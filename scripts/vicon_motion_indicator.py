"""
Minimal Vicon UDP motion indicator for latency checks.

This script listens directly to the Vicon UDP Object Stream and shows a single
large indicator window. The indicator turns red as soon as the selected object's
position moves more than a chosen threshold away from its armed baseline pose.

Use this to compare physical robot motion with the moment the laptop first sees
that motion in the raw UDP stream, without the extra rendering overhead of the
3D viewers.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from typing import Deque

from vicon_udp_viewer import parse_vicon_packet


DEFAULT_OBJECT_NAME = "jetbot"
DEFAULT_VICON_SOURCE_IP = "192.168.0.62"


@dataclass
class PoseSample:
    frame_number: int
    tx: float
    ty: float
    tz: float


class MotionIndicatorState:
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self._lock = threading.Lock()
        self._baseline: PoseSample | None = None
        self._current: PoseSample | None = None
        self._delta = 0.0
        self._triggered = False
        self._armed_time: float | None = None
        self._trigger_time: float | None = None
        self._last_packet_time: float | None = None
        self._last_sender: tuple[str, int] | None = None
        self._packet_times: Deque[float] = deque(maxlen=120)

    def update_pose(self, sample: PoseSample, sender: tuple[str, int]) -> None:
        now = time.time()

        with self._lock:
            self._current = sample
            self._last_packet_time = now
            self._last_sender = sender
            self._packet_times.append(now)

            if self._baseline is None:
                self._baseline = sample
                self._armed_time = now
                self._trigger_time = None
                self._triggered = False
                self._delta = 0.0
                return

            dx = sample.tx - self._baseline.tx
            dy = sample.ty - self._baseline.ty
            dz = sample.tz - self._baseline.tz
            self._delta = (dx * dx + dy * dy + dz * dz) ** 0.5

            if not self._triggered and self._delta >= self._threshold:
                self._triggered = True
                self._trigger_time = now
                print(
                    (
                        f"[{time.strftime('%H:%M:%S')}] Motion detected "
                        f"frame={sample.frame_number} delta={self._delta:.3f}"
                    ),
                    flush=True,
                )

    def rearm(self) -> bool:
        now = time.time()
        with self._lock:
            if self._current is None:
                return False
            self._baseline = self._current
            self._delta = 0.0
            self._triggered = False
            self._armed_time = now
            self._trigger_time = None
            return True

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            packet_rate_hz = 0.0
            if len(self._packet_times) >= 2:
                elapsed = self._packet_times[-1] - self._packet_times[0]
                if elapsed > 0.0:
                    packet_rate_hz = (len(self._packet_times) - 1) / elapsed

            return {
                "baseline": self._baseline,
                "current": self._current,
                "delta": self._delta,
                "threshold": self._threshold,
                "triggered": self._triggered,
                "armed_time": self._armed_time,
                "trigger_time": self._trigger_time,
                "last_packet_time": self._last_packet_time,
                "last_sender": self._last_sender,
                "packet_rate_hz": packet_rate_hz,
            }


class MotionReceiver(threading.Thread):
    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        source_ip: str | None,
        object_name: str,
        buffer_size: int,
        state: MotionIndicatorState,
        verbose: bool,
    ) -> None:
        super().__init__(daemon=True)
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._source_ip = source_ip
        self._object_name = object_name
        self._buffer_size = buffer_size
        self._state = state
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
        sock.settimeout(0.10)
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
                    print(f"Failed to parse packet from {sender}: {exc}", flush=True)
                continue

            for pose in poses:
                if pose.name != self._object_name:
                    continue
                sample = PoseSample(
                    frame_number=frame_number,
                    tx=pose.tx,
                    ty=pose.ty,
                    tz=pose.tz,
                )
                self._state.update_pose(sample, sender)
                break


class IndicatorApp:
    def __init__(self, root: tk.Tk, state: MotionIndicatorState, object_name: str) -> None:
        self._root = root
        self._state = state
        self._object_name = object_name

        root.title("Vicon Motion Indicator")
        root.geometry("540x520")
        root.minsize(500, 480)

        title = tk.Label(root, text="Vicon Motion Indicator", font=("Segoe UI", 18, "bold"))
        title.pack(pady=(14, 6))

        subtitle = tk.Label(
            root,
            text=f"Selected object: {object_name}",
            font=("Segoe UI", 11),
        )
        subtitle.pack()

        self._canvas = tk.Canvas(root, width=240, height=240, highlightthickness=0)
        self._canvas.pack(pady=(16, 8))
        self._oval = self._canvas.create_oval(20, 20, 220, 220, fill="#bfc9ca", outline="#7f8c8d", width=6)

        self._status_label = tk.Label(root, text="Waiting for pose...", font=("Segoe UI", 16, "bold"))
        self._status_label.pack(pady=(4, 8))

        self._info_label = tk.Label(
            root,
            text="",
            font=("Consolas", 10),
            justify="left",
            anchor="w",
        )
        self._info_label.pack(fill="x", padx=18)

        button_row = tk.Frame(root)
        button_row.pack(pady=(18, 8))

        rearm_button = tk.Button(
            button_row,
            text="Re-arm Here",
            width=14,
            command=self._rearm,
            bg="#aed6f1",
            activebackground="#d6eaf8",
        )
        rearm_button.pack(side="left", padx=8)

        quit_button = tk.Button(
            button_row,
            text="Quit",
            width=10,
            command=root.destroy,
            bg="#f5b7b1",
            activebackground="#fadbd8",
        )
        quit_button.pack(side="left", padx=8)

        hint = tk.Label(
            root,
            text="Press R to re-arm on the current pose. The red indicator latches until re-armed.",
            font=("Segoe UI", 10),
        )
        hint.pack(pady=(0, 10))

        root.bind("r", lambda _event: self._rearm())
        root.bind("R", lambda _event: self._rearm())

        self._refresh()

    def _format_pose(self, pose: PoseSample | None) -> str:
        if pose is None:
            return "(n/a, n/a, n/a)"
        return f"({pose.tx:.3f}, {pose.ty:.3f}, {pose.tz:.3f})"

    def _rearm(self) -> None:
        ok = self._state.rearm()
        if ok:
            print(f"[{time.strftime('%H:%M:%S')}] Re-armed on current pose.", flush=True)

    def _refresh(self) -> None:
        snapshot = self._state.snapshot()
        current = snapshot["current"]
        baseline = snapshot["baseline"]
        last_packet_time = snapshot["last_packet_time"]
        last_sender = snapshot["last_sender"]
        armed_time = snapshot["armed_time"]
        trigger_time = snapshot["trigger_time"]

        if current is None:
            color = "#bfc9ca"
            outline = "#7f8c8d"
            status = "Waiting for pose"
        elif snapshot["triggered"]:
            color = "#e74c3c"
            outline = "#922b21"
            status = "MOTION DETECTED"
        else:
            color = "#27ae60"
            outline = "#1e8449"
            status = "Armed"

        self._canvas.itemconfig(self._oval, fill=color, outline=outline)
        self._status_label.config(text=status)

        packet_age = None if last_packet_time is None else (time.time() - last_packet_time)
        arm_age = None if armed_time is None else (time.time() - armed_time)
        trigger_age = None if trigger_time is None else (time.time() - trigger_time)

        lines = [
            f"Threshold: {snapshot['threshold']:.3f}",
            f"Displacement: {snapshot['delta']:.3f}",
            f"UDP rate: {snapshot['packet_rate_hz']:.1f} Hz",
            f"Last packet age: {packet_age:.3f} s" if packet_age is not None else "Last packet age: n/a",
            f"Current frame: {current.frame_number}" if current is not None else "Current frame: n/a",
            f"Current pose:  {self._format_pose(current)}",
            f"Baseline pose: {self._format_pose(baseline)}",
            f"Armed age: {arm_age:.3f} s" if arm_age is not None else "Armed age: n/a",
            (
                f"Trigger age: {trigger_age:.3f} s"
                if trigger_age is not None and snapshot["triggered"]
                else "Trigger age: n/a"
            ),
            (
                f"Last sender: {last_sender[0]}:{last_sender[1]}"
                if last_sender is not None
                else "Last sender: n/a"
            ),
        ]
        self._info_label.config(text="\n".join(lines))

        self._root.after(10, self._refresh)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Light a red indicator as soon as the selected Vicon object moves past a threshold."
    )
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--source-ip", default=DEFAULT_VICON_SOURCE_IP)
    parser.add_argument("--port", type=int, default=51001)
    parser.add_argument("--object-name", default=DEFAULT_OBJECT_NAME)
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Motion threshold in Vicon translation units, which are usually millimeters.",
    )
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()

    state = MotionIndicatorState(threshold=args.threshold)
    receiver = MotionReceiver(
        bind_host=args.bind_host,
        bind_port=args.port,
        source_ip=args.source_ip or None,
        object_name=args.object_name,
        buffer_size=args.buffer_size,
        state=state,
        verbose=args.verbose,
    )
    receiver.start()

    root = tk.Tk()
    app = IndicatorApp(root, state, args.object_name)
    try:
        root.mainloop()
    finally:
        receiver.stop()
        receiver.join(timeout=1.0)
        del app


if __name__ == "__main__":
    main()
