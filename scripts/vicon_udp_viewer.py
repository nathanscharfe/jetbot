"""
Live 3D viewer for the Vicon Tracker UDP Object Stream.

This script is intended to run on your laptop while you manually drive the JetBot
from a separate notebook or interface. It listens for Vicon UDP packets, parses
tracked object poses, and displays them in real time in a 3D plot.

Expected Vicon Tracker settings:
- UDP Object Stream: enabled
- Data block size: 256, 512, or 1024
- Object Per Port: either enabled or disabled

Packet layout is based on the Vicon Tracker User Guide UDP Object Stream example:
- bytes 0-3: frame number (uint32)
- byte 4: items in block (uint8)
- repeated items:
  - byte 0: item id (uint8, 0 for object data)
  - bytes 1-2: item data size (uint16)
  - bytes 3-26: object name (24-byte null-padded string)
  - bytes 27-74: six float64 values:
    TransX, TransY, TransZ, RotX, RotY, RotZ

Run without --object-name first if you want to discover the object name being
streamed for the robot.
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


OBJECT_ITEM_ID = 0
OBJECT_NAME_BYTES = 24
OBJECT_DATA_SIZE = 72


@dataclass
class ViconPose:
    frame_number: int
    name: str
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float


@dataclass
class ObjectTrack:
    pose: ViconPose | None = None
    last_seen_time: float = 0.0
    points: Deque[tuple[float, float, float]] = field(default_factory=deque)


class SharedState:
    def __init__(self, history_size: int, stale_after_seconds: float) -> None:
        self._history_size = history_size
        self._stale_after_seconds = stale_after_seconds
        self._lock = threading.Lock()
        self._tracks: dict[str, ObjectTrack] = {}
        self._frame_number = -1
        self._packet_times: Deque[float] = deque(maxlen=120)
        self._last_packet_time: float | None = None
        self._last_sender: tuple[str, int] | None = None

    def update(
        self,
        frame_number: int,
        poses: list[ViconPose],
        sender: tuple[str, int],
    ) -> None:
        now = time.time()

        with self._lock:
            self._frame_number = frame_number
            self._packet_times.append(now)
            self._last_packet_time = now
            self._last_sender = sender

            for pose in poses:
                track = self._tracks.setdefault(pose.name, ObjectTrack())
                track.pose = pose
                track.last_seen_time = now
                if track.points.maxlen != self._history_size:
                    track.points = deque(track.points, maxlen=self._history_size)
                track.points.append((pose.tx, pose.ty, pose.tz))

    def snapshot(self, selected_name: str | None) -> dict:
        now = time.time()

        with self._lock:
            visible: dict[str, dict] = {}
            for name, track in self._tracks.items():
                if track.pose is None:
                    continue
                age = now - track.last_seen_time
                if age > self._stale_after_seconds:
                    continue
                if selected_name and name != selected_name:
                    continue
                visible[name] = {
                    "pose": track.pose,
                    "age_seconds": age,
                    "points": list(track.points),
                }

            packet_rate_hz = 0.0
            if len(self._packet_times) >= 2:
                elapsed = self._packet_times[-1] - self._packet_times[0]
                if elapsed > 0:
                    packet_rate_hz = (len(self._packet_times) - 1) / elapsed

            return {
                "frame_number": self._frame_number,
                "last_packet_time": self._last_packet_time,
                "last_sender": self._last_sender,
                "packet_rate_hz": packet_rate_hz,
                "visible": visible,
            }


def parse_vicon_packet(packet: bytes) -> tuple[int, list[ViconPose]]:
    if len(packet) < 5:
        raise ValueError(f"Packet too short: {len(packet)} bytes")

    frame_number = struct.unpack_from("<I", packet, 0)[0]
    items_in_block = packet[4]
    offset = 5
    poses: list[ViconPose] = []

    for index in range(items_in_block):
        if offset + 3 > len(packet):
            break

        item_id = packet[offset]
        item_data_size = struct.unpack_from("<H", packet, offset + 1)[0]
        offset += 3

        if offset + item_data_size > len(packet):
            break

        payload = packet[offset : offset + item_data_size]
        offset += item_data_size

        if item_id != OBJECT_ITEM_ID:
            continue
        if item_data_size < OBJECT_DATA_SIZE:
            continue

        name_bytes = payload[:OBJECT_NAME_BYTES]
        name = name_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
        if not name:
            name = f"object_{index}"

        tx, ty, tz, rx, ry, rz = struct.unpack_from("<6d", payload, OBJECT_NAME_BYTES)
        poses.append(
            ViconPose(
                frame_number=frame_number,
                name=name,
                tx=tx,
                ty=ty,
                tz=tz,
                rx=rx,
                ry=ry,
                rz=rz,
            )
        )

    return frame_number, poses


class ViconReceiver(threading.Thread):
    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        source_ip: str | None,
        shared_state: SharedState,
        buffer_size: int,
        verbose: bool,
    ) -> None:
        super().__init__(daemon=True)
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._source_ip = source_ip
        self._shared_state = shared_state
        self._buffer_size = buffer_size
        self._verbose = verbose
        self._stop_event = threading.Event()
        self._socket: socket.socket | None = None

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._bind_host, self._bind_port))
        sock.settimeout(0.25)
        self._socket = sock

        while not self._stop_event.is_set():
            try:
                packet, sender = sock.recvfrom(self._buffer_size)
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                raise

            if self._source_ip is not None and sender[0] != self._source_ip:
                continue

            try:
                frame_number, poses = parse_vicon_packet(packet)
            except Exception as exc:
                if self._verbose:
                    print(f"Failed to parse packet from {sender}: {exc}")
                continue

            self._shared_state.update(frame_number, poses, sender)


def set_axes_equal(ax, points: list[tuple[float, float, float]], padding: float) -> None:
    if not points:
        radius = max(padding, 1.0)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(-radius, radius)
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    z_center = (z_min + z_max) / 2.0

    half_span = max(
        (x_max - x_min) / 2.0,
        (y_max - y_min) / 2.0,
        (z_max - z_min) / 2.0,
        padding,
    )

    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(y_center - half_span, y_center + half_span)
    ax.set_zlim(z_center - half_span, z_center + half_span)


def build_status_lines(
    snapshot: dict,
    selected_name: str | None,
    units: str,
    display_scale: float,
) -> list[str]:
    visible = snapshot["visible"]
    lines = [
        f"Frame: {snapshot['frame_number']}",
        f"Visible objects: {len(visible)}",
        f"UDP rate: {snapshot['packet_rate_hz']:.1f} Hz",
    ]

    last_packet_time = snapshot["last_packet_time"]
    if last_packet_time is not None:
        lines.append(f"Last packet age: {time.time() - last_packet_time:.2f} s")

    last_sender = snapshot["last_sender"]
    if last_sender is not None:
        lines.append(f"Last sender: {last_sender[0]}:{last_sender[1]}")

    if selected_name:
        lines.append(f"Selected object: {selected_name}")

    if len(visible) == 1:
        pose = next(iter(visible.values()))["pose"]
        lines.extend(
            [
                (
                    f"Pos ({units}): ("
                    f"{pose.tx * display_scale:.3f}, "
                    f"{pose.ty * display_scale:.3f}, "
                    f"{pose.tz * display_scale:.3f})"
                ),
                f"Rot (Euler XYZ): ({pose.rx:.3f}, {pose.ry:.3f}, {pose.rz:.3f})",
            ]
        )

    return lines


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Listen to the Vicon Tracker UDP stream and show tracked objects in 3D."
    )
    parser.add_argument(
        "--bind-host",
        default="0.0.0.0",
        help="Local interface to bind to. Use 0.0.0.0 to listen on all interfaces.",
    )
    parser.add_argument(
        "--source-ip",
        default="192.168.0.62",
        help="Optional sender IP filter for the Vicon UDP stream.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=51001,
        help="UDP port to listen on. Match this to the Vicon Tracker UDP stream port.",
    )
    parser.add_argument(
        "--object-name",
        default="jetbot",
        help="Exact Vicon object name to display. Use an empty string to show all active objects.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=200,
        help="Number of historical samples to keep per object for the trail.",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=2.0,
        help="Hide an object if it has not been updated for this many seconds.",
    )
    parser.add_argument(
        "--update-interval-ms",
        type=int,
        default=100,
        help="Plot refresh interval in milliseconds.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=4096,
        help="UDP receive buffer size in bytes.",
    )
    parser.add_argument(
        "--units",
        choices=("mm", "m"),
        default="mm",
        help="Display translation units as millimeters or meters.",
    )
    parser.add_argument(
        "--axis-padding",
        type=float,
        default=250.0,
        help="Minimum half-span of the plot view in Vicon translation units (millimeters).",
    )
    parser.add_argument(
        "--elevation-deg",
        type=float,
        default=25.0,
        help="3D camera elevation angle for the plot.",
    )
    parser.add_argument(
        "--azimuth-deg",
        type=float,
        default=45.0,
        help="3D camera azimuth angle for the plot.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print parse errors for malformed or unexpected packets.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    selected_object_name = args.object_name or None

    display_scale = 1.0 if args.units == "mm" else 0.001
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
    receiver.start()

    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(right=0.8, top=0.92)
    ax = fig.add_subplot(111, projection="3d")
    color_cycle = list(plt.get_cmap("tab10").colors)

    def update_plot(_frame_index: int) -> None:
        snapshot = shared_state.snapshot(selected_object_name)
        visible = snapshot["visible"]

        ax.cla()
        ax.set_xlabel(f"X ({args.units})")
        ax.set_ylabel(f"Y ({args.units})")
        ax.set_zlabel(f"Z ({args.units})")
        ax.set_title("Vicon UDP 3D Viewer")
        ax.view_init(elev=args.elevation_deg, azim=args.azimuth_deg)

        plotted_points: list[tuple[float, float, float]] = []
        names = sorted(visible)

        for index, name in enumerate(names):
            color = color_cycle[index % len(color_cycle)]
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

            x = pose.tx * display_scale
            y = pose.ty * display_scale
            z = pose.tz * display_scale
            ax.scatter([x], [y], [z], color=[color], s=60, label=name)
            ax.text(x, y, z, f" {name}", color=color)
            plotted_points.append((x, y, z))

        ax.scatter([0.0], [0.0], [0.0], color="black", marker="x", s=40)

        set_axes_equal(ax, plotted_points, padding=args.axis_padding * display_scale)

        if names and len(names) <= 10:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

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
        receiver.stop()
        receiver.join(timeout=1.0)
        del animation


if __name__ == "__main__":
    main()
