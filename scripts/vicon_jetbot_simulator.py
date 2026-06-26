"""
Laptop-side simulator for the Vicon UDP stream and JetBot TCP control link.

This lets the MPPI viewer/controller scripts run without the real Vicon system
or JetBot hardware:
- emits example Vicon UDP Object Stream packets
- accepts JetBot-style newline-delimited JSON drive commands over TCP
- ignores those commands for now and keeps publishing scripted example poses

Typical use with the simulator running on the same laptop:

    python scripts/vicon_jetbot_simulator.py

Then launch the controller against loopback addresses, for example:

    python scripts/vicon_goto_mppi_obstacle_controller.py --source-ip 127.0.0.1 --jetbot-host 127.0.0.1
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass


OBJECT_ITEM_ID = 0
OBJECT_NAME_BYTES = 24

DEFAULT_OBJECT_NAME = "jetbot"
DEFAULT_UDP_SOURCE_HOST = "127.0.0.1"
DEFAULT_UDP_DEST_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 51001
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 8765


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class SimPose:
    name: str
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float


def pack_object_payload(pose: SimPose) -> bytes:
    name_bytes = pose.name.encode("ascii", errors="ignore")[:OBJECT_NAME_BYTES]
    name_bytes = name_bytes.ljust(OBJECT_NAME_BYTES, b"\x00")
    transform_bytes = struct.pack("<6d", pose.tx, pose.ty, pose.tz, pose.rx, pose.ry, pose.rz)
    return name_bytes + transform_bytes


def build_vicon_packet(frame_number: int, poses: list[SimPose]) -> bytes:
    packet = bytearray()
    packet.extend(struct.pack("<I", frame_number))
    packet.append(len(poses) & 0xFF)

    for pose in poses:
        payload = pack_object_payload(pose)
        packet.append(OBJECT_ITEM_ID)
        packet.extend(struct.pack("<H", len(payload)))
        packet.extend(payload)

    return bytes(packet)


class ExampleViconStreamer(threading.Thread):
    def __init__(
        self,
        *,
        udp_source_host: str,
        udp_dest_host: str,
        udp_port: int,
        object_name: str,
        packet_rate_hz: float,
        verbose: bool,
    ) -> None:
        super().__init__(daemon=True)
        self._udp_source_host = udp_source_host
        self._udp_dest_host = udp_dest_host
        self._udp_port = udp_port
        self._object_name = object_name
        self._packet_period = 1.0 / max(packet_rate_hz, 1.0)
        self._verbose = verbose
        self._stop_event = threading.Event()
        self._socket: socket.socket | None = None
        self._frame_number = 0
        self._start_time = time.time()

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass

    def _scripted_poses(self, now: float) -> list[SimPose]:
        elapsed = now - self._start_time
        angle = 2.0 * math.pi * elapsed / 12.0
        tx = 650.0 * math.cos(angle)
        ty = 450.0 * math.sin(angle)
        tz = 130.0
        rz = angle + math.pi / 2.0

        robot_pose = SimPose(
            name=self._object_name,
            tx=tx,
            ty=ty,
            tz=tz,
            rx=0.0,
            ry=0.0,
            rz=rz,
        )
        obstacle_a = SimPose(
            name="obstacle_a",
            tx=700.0,
            ty=250.0,
            tz=120.0,
            rx=0.0,
            ry=0.0,
            rz=0.0,
        )
        obstacle_b = SimPose(
            name="obstacle_b",
            tx=-850.0,
            ty=-500.0,
            tz=120.0,
            rx=0.0,
            ry=0.0,
            rz=0.0,
        )
        obstacle_c = SimPose(
            name="obstacle_c",
            tx=350.0 * math.cos(angle * 0.5 + 0.8) - 200.0,
            ty=300.0 * math.sin(angle * 0.5 + 0.8) + 800.0,
            tz=120.0,
            rx=0.0,
            ry=0.0,
            rz=0.0,
        )
        return [robot_pose, obstacle_a, obstacle_b, obstacle_c]

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self._udp_dest_host.endswith(".255") or self._udp_dest_host == "255.255.255.255":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((self._udp_source_host, 0))
        self._socket = sock

        if self._verbose:
            print(
                f"UDP simulator sending from {self._udp_source_host} to "
                f"{self._udp_dest_host}:{self._udp_port}"
            )

        next_send_time = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            poses = self._scripted_poses(now)
            packet = build_vicon_packet(self._frame_number, poses)
            try:
                sock.sendto(packet, (self._udp_dest_host, self._udp_port))
            except OSError:
                if self._stop_event.is_set():
                    break
                raise

            self._frame_number += 1
            next_send_time += self._packet_period
            sleep_time = next_send_time - time.time()
            if sleep_time > 0.0:
                self._stop_event.wait(sleep_time)
            else:
                next_send_time = time.time()


class DummyJetBotState:
    def __init__(self, *, verbose: bool) -> None:
        self._verbose = verbose
        self._lock = threading.Lock()
        self._last_message: dict | None = None
        self._last_client = ""
        self._drive_count = 0
        self._stop_count = 0
        self._ping_count = 0

    def handle_message(self, client_label: str, message: dict) -> None:
        message_type = str(message.get("type", ""))
        with self._lock:
            self._last_message = dict(message)
            self._last_client = client_label
            if message_type == "drive":
                self._drive_count += 1
            elif message_type == "stop":
                self._stop_count += 1
            elif message_type == "ping":
                self._ping_count += 1

        if self._verbose:
            print(f"TCP {client_label} -> {message}")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "last_message": dict(self._last_message) if self._last_message is not None else None,
                "last_client": self._last_client,
                "drive_count": self._drive_count,
                "stop_count": self._stop_count,
                "ping_count": self._ping_count,
            }


class DummyJetBotRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        client_label = f"{self.client_address[0]}:{self.client_address[1]}"
        if self.server.sim_verbose:
            print(f"TCP client connected: {client_label}")

        try:
            while True:
                raw_line = self.rfile.readline()
                if not raw_line:
                    break

                try:
                    message = json.loads(raw_line.decode("utf-8").strip())
                except json.JSONDecodeError as exc:
                    if self.server.sim_verbose:
                        print(f"Bad JSON from {client_label}: {exc}")
                    continue

                self.server.sim_state.handle_message(client_label, message)
        except Exception as exc:
            if self.server.sim_verbose:
                print(f"TCP client error from {client_label}: {exc}")
        finally:
            if self.server.sim_verbose:
                print(f"TCP client disconnected: {client_label}")


class DummyJetBotTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, request_handler_class, sim_state: DummyJetBotState, sim_verbose: bool) -> None:
        super().__init__(server_address, request_handler_class)
        self.sim_state = sim_state
        self.sim_verbose = sim_verbose


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate the Vicon UDP pose stream and JetBot TCP control socket on a laptop."
    )
    parser.add_argument("--udp-source-host", default=DEFAULT_UDP_SOURCE_HOST)
    parser.add_argument("--udp-dest-host", default=DEFAULT_UDP_DEST_HOST)
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    parser.add_argument("--tcp-host", default=DEFAULT_TCP_HOST)
    parser.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    parser.add_argument("--object-name", default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--packet-rate-hz", type=float, default=60.0)
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="Optional finite run time for smoke tests. Use 0 to run until Ctrl+C.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    tcp_state = DummyJetBotState(verbose=args.verbose)
    tcp_server = DummyJetBotTCPServer(
        (args.tcp_host, args.tcp_port),
        DummyJetBotRequestHandler,
        sim_state=tcp_state,
        sim_verbose=args.verbose,
    )
    tcp_thread = threading.Thread(target=tcp_server.serve_forever, daemon=True)
    udp_streamer = ExampleViconStreamer(
        udp_source_host=args.udp_source_host,
        udp_dest_host=args.udp_dest_host,
        udp_port=args.udp_port,
        object_name=args.object_name,
        packet_rate_hz=args.packet_rate_hz,
        verbose=args.verbose,
    )

    print(
        f"Simulator ready: UDP {args.udp_source_host} -> {args.udp_dest_host}:{args.udp_port}, "
        f"TCP {args.tcp_host}:{args.tcp_port}, object '{args.object_name}'"
    )
    print("Use Ctrl+C to stop. TCP commands are accepted but ignored for motion right now.")

    tcp_thread.start()
    udp_streamer.start()

    end_time = time.time() + args.run_seconds if args.run_seconds > 0.0 else None
    try:
        while True:
            if end_time is not None and time.time() >= end_time:
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        udp_streamer.stop()
        tcp_server.shutdown()
        tcp_server.server_close()
        udp_streamer.join(timeout=1.0)

        snapshot = tcp_state.snapshot()
        if snapshot["last_message"] is not None:
            print(
                "Last TCP command:",
                snapshot["last_message"],
                f"from {snapshot['last_client']}",
            )
        print(
            f"TCP counts: drive={snapshot['drive_count']}, "
            f"stop={snapshot['stop_count']}, ping={snapshot['ping_count']}"
        )


if __name__ == "__main__":
    main()
