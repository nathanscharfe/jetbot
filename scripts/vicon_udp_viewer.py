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
import math
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> list[list[float]]:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ]
    ry_matrix = [
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ]
    rz_matrix = [
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ]

    def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [
            [
                sum(a[row][k] * b[k][col] for k in range(3))
                for col in range(3)
            ]
            for row in range(3)
        ]

    return matmul(rz_matrix, matmul(ry_matrix, rx_matrix))


def rotate_vector(
    rotation: list[list[float]],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        rotation[0][0] * vector[0] + rotation[0][1] * vector[1] + rotation[0][2] * vector[2],
        rotation[1][0] * vector[0] + rotation[1][1] * vector[1] + rotation[1][2] * vector[2],
        rotation[2][0] * vector[0] + rotation[2][1] * vector[1] + rotation[2][2] * vector[2],
    )


def add_vector(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def normalize_xy(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    magnitude = math.hypot(vector[0], vector[1])
    if magnitude <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (vector[0] / magnitude, vector[1] / magnitude, 0.0)


def room_corners(
    room_center: tuple[float, float, float],
    room_size: tuple[float, float, float],
) -> dict[str, tuple[float, float, float]]:
    cx, cy, cz = room_center
    sx, sy, sz = room_size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    return {
        "lbf": (cx - hx, cy - hy, cz - hz),
        "lbb": (cx - hx, cy + hy, cz - hz),
        "rbf": (cx + hx, cy - hy, cz - hz),
        "rbb": (cx + hx, cy + hy, cz - hz),
        "ltf": (cx - hx, cy - hy, cz + hz),
        "ltb": (cx - hx, cy + hy, cz + hz),
        "rtf": (cx + hx, cy - hy, cz + hz),
        "rtb": (cx + hx, cy + hy, cz + hz),
    }


def draw_room(
    ax,
    room_center: tuple[float, float, float],
    room_size: tuple[float, float, float],
    units: str,
) -> list[tuple[float, float, float]]:
    corners = room_corners(room_center, room_size)
    floor = [corners["lbf"], corners["rbf"], corners["rbb"], corners["lbb"]]
    ceiling = [corners["ltf"], corners["rtf"], corners["rtb"], corners["ltb"]]
    wall_loops = [
        [corners["lbf"], corners["rbf"], corners["rtf"], corners["ltf"]],
        [corners["rbf"], corners["rbb"], corners["rtb"], corners["rtf"]],
        [corners["rbb"], corners["lbb"], corners["ltb"], corners["rtb"]],
        [corners["lbb"], corners["lbf"], corners["ltf"], corners["ltb"]],
    ]

    floor_patch = Poly3DCollection([floor], alpha=0.08, facecolor="#7fb3d5", edgecolor="none")
    ceiling_patch = Poly3DCollection([ceiling], alpha=0.02, facecolor="#d5dbdb", edgecolor="none")
    ax.add_collection3d(floor_patch)
    ax.add_collection3d(ceiling_patch)

    edge_segments = [
        ("lbf", "rbf"),
        ("rbf", "rbb"),
        ("rbb", "lbb"),
        ("lbb", "lbf"),
        ("ltf", "rtf"),
        ("rtf", "rtb"),
        ("rtb", "ltb"),
        ("ltb", "ltf"),
        ("lbf", "ltf"),
        ("rbf", "rtf"),
        ("rbb", "rtb"),
        ("lbb", "ltb"),
    ]
    for start_key, end_key in edge_segments:
        start = corners[start_key]
        end = corners[end_key]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="#7f8c8d",
            linewidth=1.0,
            alpha=0.55,
        )

    for wall in wall_loops:
        loop = wall + [wall[0]]
        ax.plot(
            [point[0] for point in loop],
            [point[1] for point in loop],
            [point[2] for point in loop],
            color="#bdc3c7",
            linewidth=0.8,
            alpha=0.25,
        )

    floor_z = floor[0][2]
    x_ticks = 4
    y_ticks = 4
    x_min, x_max = corners["lbf"][0], corners["rbf"][0]
    y_min, y_max = corners["lbf"][1], corners["lbb"][1]

    for index in range(1, x_ticks):
        x = x_min + (x_max - x_min) * index / x_ticks
        ax.plot(
            [x, x],
            [y_min, y_max],
            [floor_z, floor_z],
            color="#d5d8dc",
            linewidth=0.8,
            alpha=0.4,
        )
    for index in range(1, y_ticks):
        y = y_min + (y_max - y_min) * index / y_ticks
        ax.plot(
            [x_min, x_max],
            [y, y],
            [floor_z, floor_z],
            color="#d5d8dc",
            linewidth=0.8,
            alpha=0.4,
        )

    ax.text(
        corners["lbf"][0],
        corners["lbf"][1],
        corners["ltf"][2],
        f"Room ({units})",
        color="#566573",
        fontsize=9,
    )

    return list(corners.values())


def draw_pose_marker(
    ax,
    pose: ViconPose,
    display_scale: float,
    color,
    label: str,
    body_size: float,
    axis_length: float,
) -> list[tuple[float, float, float]]:
    center = (
        pose.tx * display_scale,
        pose.ty * display_scale,
        pose.tz * display_scale,
    )
    rotation = rotation_matrix_xyz(pose.rx, pose.ry, pose.rz)

    half_body = body_size / 2.0
    local_corners = [
        (-half_body, -half_body, 0.0),
        (half_body, -half_body, 0.0),
        (half_body, half_body, 0.0),
        (-half_body, half_body, 0.0),
    ]
    world_corners = [
        add_vector(center, rotate_vector(rotation, corner))
        for corner in local_corners
    ]
    loop = world_corners + [world_corners[0]]
    ax.plot(
        [point[0] for point in loop],
        [point[1] for point in loop],
        [point[2] for point in loop],
        color=color,
        linewidth=1.8,
    )

    forward_tip = add_vector(center, rotate_vector(rotation, (axis_length, 0.0, 0.0)))
    left_tip = add_vector(center, rotate_vector(rotation, (0.0, axis_length * 0.7, 0.0)))
    up_tip = add_vector(center, rotate_vector(rotation, (0.0, 0.0, axis_length * 0.7)))
    # The Vicon rigid-body definition for this JetBot uses body-Y (green) as physical forward.
    physical_forward_tip = add_vector(center, rotate_vector(rotation, (0.0, axis_length, 0.0)))
    horizontal_heading = normalize_xy(
        (
            physical_forward_tip[0] - center[0],
            physical_forward_tip[1] - center[1],
            physical_forward_tip[2] - center[2],
        )
    )
    heading_tip = add_vector(
        center,
        (
            horizontal_heading[0] * axis_length,
            horizontal_heading[1] * axis_length,
            0.0,
        ),
    )

    ax.quiver(
        center[0],
        center[1],
        center[2],
        forward_tip[0] - center[0],
        forward_tip[1] - center[1],
        forward_tip[2] - center[2],
        color="#e74c3c",
        linewidth=2.0,
        arrow_length_ratio=0.2,
    )
    ax.quiver(
        center[0],
        center[1],
        center[2],
        left_tip[0] - center[0],
        left_tip[1] - center[1],
        left_tip[2] - center[2],
        color="#27ae60",
        linewidth=1.6,
        arrow_length_ratio=0.2,
    )
    ax.quiver(
        center[0],
        center[1],
        center[2],
        up_tip[0] - center[0],
        up_tip[1] - center[1],
        up_tip[2] - center[2],
        color="#2980b9",
        linewidth=1.6,
        arrow_length_ratio=0.2,
    )
    ax.quiver(
        center[0],
        center[1],
        center[2],
        heading_tip[0] - center[0],
        heading_tip[1] - center[1],
        0.0,
        color="#111111",
        linewidth=2.8,
        arrow_length_ratio=0.18,
        alpha=0.95,
    )
    ax.scatter([center[0]], [center[1]], [center[2]], color=[color], s=55, zorder=5)
    ax.text(center[0], center[1], center[2], f" {label}", color=color)

    return world_corners + [forward_tip, left_tip, up_tip, physical_forward_tip, heading_tip, center]


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
        "--room-size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(4000.0, 4000.0, 2500.0),
        help="Displayed room size in Vicon translation units, defaulting to a 4m x 4m x 2.5m room.",
    )
    parser.add_argument(
        "--room-center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 1250.0),
        help="Displayed room center in Vicon translation units.",
    )
    parser.add_argument(
        "--robot-footprint",
        type=float,
        default=250.0,
        help="Displayed robot body width in Vicon translation units.",
    )
    parser.add_argument(
        "--robot-axis-length",
        type=float,
        default=350.0,
        help="Displayed robot heading axis length in Vicon translation units.",
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
