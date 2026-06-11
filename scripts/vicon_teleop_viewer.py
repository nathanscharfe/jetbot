"""
Combined Vicon 3D viewer and JetBot teleoperation client.

Run this on the laptop. It:
- receives the Vicon Tracker UDP Object Stream
- shows the selected object live in 3D
- sends left/right motor commands to a small socket server running on the JetBot

Keyboard controls:
- W / Up: forward
- S / Down: reverse
- A / Left: turn left
- D / Right: turn right
- Space / X / Escape: stop

Mouse controls:
- Press and hold the on-screen Forward / Reverse / Left / Right buttons
- Release the mouse button to stop that motion
- Click Stop at any time to force zero motor command

This script intentionally does not drive the motors directly. It only sends
commands to a server running on the JetBot.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button

from vicon_udp_viewer import (
    SharedState,
    ViconReceiver,
    build_status_lines,
    draw_pose_marker,
    draw_room,
    set_axes_equal,
)


DEFAULT_VICON_SOURCE_IP = "192.168.0.62"
DEFAULT_OBJECT_NAME = "jetbot"
DEFAULT_JETBOT_HOST = "10.53.174.144"
DEFAULT_JETBOT_PORT = 8765
MOTION_KEYS = {"w", "a", "s", "d", "up", "down", "left", "right"}


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class DriveCommand:
    left: float = 0.0
    right: float = 0.0
    updated_time: float = 0.0


class DriveState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._command = DriveCommand(updated_time=time.time())

    def set(self, left: float, right: float) -> None:
        with self._lock:
            self._command = DriveCommand(
                left=clamp(left),
                right=clamp(right),
                updated_time=time.time(),
            )

    def stop(self) -> None:
        self.set(0.0, 0.0)

    def snapshot(self) -> DriveCommand:
        with self._lock:
            return DriveCommand(
                left=self._command.left,
                right=self._command.right,
                updated_time=self._command.updated_time,
            )


class TeleopClient(threading.Thread):
    def __init__(
        self,
        host: str,
        port: int,
        drive_state: DriveState,
        send_rate_hz: float,
        verbose: bool,
    ) -> None:
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._drive_state = drive_state
        self._send_period = 1.0 / max(send_rate_hz, 1.0)
        self._verbose = verbose
        self._stop_event = threading.Event()
        self._socket: socket.socket | None = None
        self._status_lock = threading.Lock()
        self._connected = False
        self._last_error = ""
        self._last_send_time: float | None = None
        self._last_connect_time: float | None = None

    def stop(self) -> None:
        self._stop_event.set()
        self._send_message({"type": "stop"})
        self._close_socket()

    def snapshot(self) -> dict:
        with self._status_lock:
            return {
                "connected": self._connected,
                "last_error": self._last_error,
                "last_send_time": self._last_send_time,
                "last_connect_time": self._last_connect_time,
                "host": self._host,
                "port": self._port,
            }

    def _set_status(
        self,
        *,
        connected: bool | None = None,
        last_error: str | None = None,
        last_send_time: float | None = None,
        last_connect_time: float | None = None,
    ) -> None:
        with self._status_lock:
            if connected is not None:
                self._connected = connected
            if last_error is not None:
                self._last_error = last_error
            if last_send_time is not None:
                self._last_send_time = last_send_time
            if last_connect_time is not None:
                self._last_connect_time = last_connect_time

    def _connect(self) -> bool:
        if self._socket is not None:
            return True

        try:
            sock = socket.create_connection((self._host, self._port), timeout=2.0)
            sock.settimeout(1.0)
        except OSError as exc:
            self._set_status(connected=False, last_error=str(exc))
            if self._verbose:
                print(f"Teleop connect failed: {exc}")
            return False

        self._socket = sock
        self._set_status(
            connected=True,
            last_error="",
            last_connect_time=time.time(),
        )
        return True

    def _close_socket(self) -> None:
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        self._set_status(connected=False)

    def _send_message(self, payload: dict) -> bool:
        if self._socket is None and not self._connect():
            return False

        data = (json.dumps(payload) + "\n").encode("utf-8")
        try:
            assert self._socket is not None
            self._socket.sendall(data)
        except OSError as exc:
            self._set_status(connected=False, last_error=str(exc))
            if self._verbose:
                print(f"Teleop send failed: {exc}")
            self._close_socket()
            return False

        self._set_status(last_send_time=time.time())
        return True

    def run(self) -> None:
        while not self._stop_event.is_set():
            command = self._drive_state.snapshot()
            self._send_message(
                {
                    "type": "drive",
                    "left": command.left,
                    "right": command.right,
                    "client_time": time.time(),
                }
            )
            self._stop_event.wait(self._send_period)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive the Vicon UDP stream, show it in 3D, and teleoperate the JetBot."
    )
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--source-ip", default=DEFAULT_VICON_SOURCE_IP)
    parser.add_argument("--port", type=int, default=51001)
    parser.add_argument("--object-name", default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--history", type=int, default=200)
    parser.add_argument("--stale-after", type=float, default=2.0)
    parser.add_argument("--update-interval-ms", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--units", choices=("mm", "m"), default="mm")
    parser.add_argument("--axis-padding", type=float, default=250.0)
    parser.add_argument("--elevation-deg", type=float, default=25.0)
    parser.add_argument("--azimuth-deg", type=float, default=45.0)
    parser.add_argument(
        "--room-size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(4000.0, 4000.0, 2500.0),
        help="Displayed room size in Vicon translation units.",
    )
    parser.add_argument(
        "--room-center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 1250.0),
        help="Displayed room center in Vicon translation units.",
    )
    parser.add_argument("--robot-footprint", type=float, default=250.0)
    parser.add_argument("--robot-axis-length", type=float, default=350.0)
    parser.add_argument("--speed", type=float, default=0.7, help="Base forward/reverse speed.")
    parser.add_argument("--turn-speed", type=float, default=0.5, help="Base turning speed.")
    parser.add_argument("--jetbot-host", default=DEFAULT_JETBOT_HOST)
    parser.add_argument("--jetbot-port", type=int, default=DEFAULT_JETBOT_PORT)
    parser.add_argument("--send-rate-hz", type=float, default=15.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    selected_object_name = args.object_name or None
    display_scale = 1.0 if args.units == "mm" else 0.001
    room_center = tuple(value * display_scale for value in args.room_center)
    room_size = tuple(value * display_scale for value in args.room_size)
    robot_footprint = args.robot_footprint * display_scale
    robot_axis_length = args.robot_axis_length * display_scale

    shared_state = SharedState(
        history_size=args.history,
        stale_after_seconds=args.stale_after,
    )
    receiver = ViconReceiver(
        bind_host=args.bind_host,
        bind_port=args.port,
        source_ip=args.source_ip or None,
        shared_state=shared_state,
        buffer_size=args.buffer_size,
        verbose=args.verbose,
    )

    drive_state = DriveState()
    teleop_client = TeleopClient(
        host=args.jetbot_host,
        port=args.jetbot_port,
        drive_state=drive_state,
        send_rate_hz=args.send_rate_hz,
        verbose=args.verbose,
    )

    active_keys: set[str] = set()
    input_lock = threading.Lock()
    active_mouse_command: tuple[float, float] | None = None

    def apply_current_command() -> None:
        nonlocal active_mouse_command

        with input_lock:
            mouse_command = active_mouse_command
            keys = set(active_keys)

        if mouse_command is not None:
            drive_state.set(*mouse_command)
            return

        forward = bool(keys & {"w", "up"})
        reverse = bool(keys & {"s", "down"})
        turn_left = bool(keys & {"a", "left"})
        turn_right = bool(keys & {"d", "right"})

        if forward and not reverse:
            throttle = args.speed
        elif reverse and not forward:
            throttle = -args.speed
        else:
            throttle = 0.0

        if turn_left and not turn_right:
            steering = args.turn_speed
        elif turn_right and not turn_left:
            steering = -args.turn_speed
        else:
            steering = 0.0

        left = clamp(throttle - steering)
        right = clamp(throttle + steering)
        drive_state.set(left, right)

    def stop_motion() -> None:
        nonlocal active_mouse_command
        with input_lock:
            active_keys.clear()
            active_mouse_command = None
        drive_state.stop()

    def on_key_press(event) -> None:
        key = (event.key or "").lower()
        if key in {" ", "space", "x", "escape"}:
            stop_motion()
            return
        if key in MOTION_KEYS:
            with input_lock:
                active_keys.add(key)
            apply_current_command()

    def on_key_release(event) -> None:
        key = (event.key or "").lower()
        if key in MOTION_KEYS:
            with input_lock:
                active_keys.discard(key)
            apply_current_command()

    receiver.start()
    teleop_client.start()

    fig = plt.figure(figsize=(11, 8))
    fig.subplots_adjust(bottom=0.26, right=0.8, top=0.92)
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    button_specs = [
        {
            "label": "Forward",
            "rect": [0.40, 0.15, 0.16, 0.05],
            "command": (args.speed, args.speed),
            "color": "#8fd19e",
        },
        {
            "label": "Left",
            "rect": [0.22, 0.08, 0.16, 0.05],
            "command": (-args.turn_speed, args.turn_speed),
            "color": "#9ec5fe",
        },
        {
            "label": "Stop",
            "rect": [0.40, 0.08, 0.16, 0.05],
            "command": None,
            "color": "#f5a3a3",
        },
        {
            "label": "Right",
            "rect": [0.58, 0.08, 0.16, 0.05],
            "command": (args.turn_speed, -args.turn_speed),
            "color": "#9ec5fe",
        },
        {
            "label": "Reverse",
            "rect": [0.40, 0.01, 0.16, 0.05],
            "command": (-args.speed, -args.speed),
            "color": "#f7c97f",
        },
    ]
    button_axes_to_command: dict[object, tuple[float, float] | None] = {}
    button_widgets: list[Button] = []

    for spec in button_specs:
        button_ax = fig.add_axes(spec["rect"])
        button = Button(button_ax, spec["label"], color=spec["color"], hovercolor="#dddddd")
        button.label.set_fontsize(10)
        button_axes_to_command[button_ax] = spec["command"]
        button_widgets.append(button)

    def on_mouse_press(event) -> None:
        nonlocal active_mouse_command
        if event.inaxes not in button_axes_to_command:
            return

        command = button_axes_to_command[event.inaxes]
        if command is None:
            stop_motion()
            return

        with input_lock:
            active_mouse_command = command
        apply_current_command()

    def on_mouse_release(_event) -> None:
        nonlocal active_mouse_command
        with input_lock:
            had_mouse_command = active_mouse_command is not None
            active_mouse_command = None
        if had_mouse_command:
            apply_current_command()

    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)

    color_cycle = list(plt.get_cmap("tab10").colors)

    def update_plot(_frame_index: int) -> None:
        snapshot = shared_state.snapshot(selected_object_name)
        visible = snapshot["visible"]
        teleop_status = teleop_client.snapshot()
        drive_command = drive_state.snapshot()

        ax.cla()
        ax.set_xlabel(f"X ({args.units})")
        ax.set_ylabel(f"Y ({args.units})")
        ax.set_zlabel(f"Z ({args.units})")
        ax.set_title("Vicon Teleop Viewer")
        ax.view_init(elev=args.elevation_deg, azim=args.azimuth_deg)

        plotted_points = draw_room(ax, room_center, room_size, args.units)
        names = sorted(visible)
        legend_handles: list[Line2D] = []

        for index, name in enumerate(names):
            color = color_cycle[index % len(color_cycle)]
            legend_handles.append(
                Line2D([0], [0], color=color, marker="o", linestyle="-", linewidth=1.5, markersize=6, label=name)
            )
            pose = visible[name]["pose"]
            history_points = [
                (x * display_scale, y * display_scale, z * display_scale)
                for x, y, z in visible[name]["points"]
            ]

            if history_points:
                hx = [point[0] for point in history_points]
                hy = [point[1] for point in history_points]
                hz = [point[2] for point in history_points]
                ax.plot(hx, hy, hz, color=color, linewidth=1.5, alpha=0.75)
                plotted_points.extend(history_points)

            plotted_points.extend(
                draw_pose_marker(
                    ax,
                    pose,
                    display_scale=display_scale,
                    color=color,
                    label=name,
                    body_size=robot_footprint,
                    axis_length=robot_axis_length,
                )
            )

        ax.scatter([0.0], [0.0], [0.0], color="black", marker="x", s=40)
        set_axes_equal(ax, plotted_points, padding=args.axis_padding * display_scale)

        if legend_handles and len(legend_handles) <= 10:
            ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        status_lines = build_status_lines(
            snapshot,
            selected_object_name,
            args.units,
            display_scale,
        )

        if snapshot["last_packet_time"] is None:
            status_lines.append("Waiting for UDP packets...")
        elif selected_object_name and not names:
            status_lines.append("Receiving packets, but the selected object is not visible.")

        status_lines.extend(
            [
                "",
                f"JetBot server: {teleop_status['host']}:{teleop_status['port']}",
                f"Teleop link: {'connected' if teleop_status['connected'] else 'disconnected'}",
                f"Drive command: ({drive_command.left:.2f}, {drive_command.right:.2f})",
                f"Speed settings: speed={args.speed:.2f}, turn={args.turn_speed:.2f}",
                "Controls: press and hold buttons below plot, Space to stop",
            ]
        )

        last_send_time = teleop_status["last_send_time"]
        if last_send_time is not None:
            status_lines.append(f"Last command age: {time.time() - last_send_time:.2f} s")

        if teleop_status["last_error"]:
            status_lines.append(f"Teleop error: {teleop_status['last_error']}")

        ax.text2D(
            0.02,
            0.98,
            "\n".join(status_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "lightgray"},
        )

    animation = FuncAnimation(
        fig,
        update_plot,
        interval=args.update_interval_ms,
        cache_frame_data=False,
    )

    try:
        plt.show()
    finally:
        stop_motion()
        teleop_client.stop()
        receiver.stop()
        teleop_client.join(timeout=1.0)
        receiver.join(timeout=1.0)
        del animation


if __name__ == "__main__":
    main()
