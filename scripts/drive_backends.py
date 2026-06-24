"""
Drive command state plus transport backends for robot teleoperation.

This module supports:
- the original JetBot JSON-over-TCP socket transport
- an Arduino Bluetooth serial transport that accepts D,<left>,<right>
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass

import serial


DEFAULT_DRIVE_BACKEND = "jetbot-socket"
DRIVE_BACKEND_CHOICES = ("jetbot-socket", "arduino-bluetooth")


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


class DriveClientBase(threading.Thread):
    def __init__(
        self,
        *,
        backend_name: str,
        transport_name: str,
        endpoint: str,
        drive_state: DriveState,
        send_rate_hz: float,
        verbose: bool,
    ) -> None:
        super().__init__(daemon=True)
        self._backend_name = backend_name
        self._transport_name = transport_name
        self._endpoint = endpoint
        self._drive_state = drive_state
        self._send_period = 1.0 / max(send_rate_hz, 1.0)
        self._verbose = verbose
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._connected = False
        self._last_error = ""
        self._last_send_time: float | None = None
        self._last_connect_time: float | None = None

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._send_stop_command()
        finally:
            self._close_connection()

    def snapshot(self) -> dict:
        with self._status_lock:
            return {
                "backend": self._backend_name,
                "transport": self._transport_name,
                "endpoint": self._endpoint,
                "connected": self._connected,
                "last_error": self._last_error,
                "last_send_time": self._last_send_time,
                "last_connect_time": self._last_connect_time,
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

    def _record_failure(self, exc: Exception, prefix: str) -> None:
        self._set_status(connected=False, last_error=str(exc))
        if self._verbose:
            print(f"{prefix}: {exc}")

    def _connect(self) -> bool:
        raise NotImplementedError

    def _close_connection(self) -> None:
        raise NotImplementedError

    def _send_stop_command(self) -> bool:
        raise NotImplementedError

    def _send_drive_command(self, command: DriveCommand) -> bool:
        raise NotImplementedError

    def run(self) -> None:
        while not self._stop_event.is_set():
            command = self._drive_state.snapshot()
            self._send_drive_command(command)
            self._stop_event.wait(self._send_period)


class JetBotSocketClient(DriveClientBase):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        drive_state: DriveState,
        send_rate_hz: float,
        verbose: bool,
    ) -> None:
        super().__init__(
            backend_name="jetbot-socket",
            transport_name="tcp",
            endpoint=f"{host}:{port}",
            drive_state=drive_state,
            send_rate_hz=send_rate_hz,
            verbose=verbose,
        )
        self._host = host
        self._port = port
        self._socket: socket.socket | None = None

    def _connect(self) -> bool:
        if self._socket is not None:
            return True

        try:
            sock = socket.create_connection((self._host, self._port), timeout=2.0)
            sock.settimeout(1.0)
        except OSError as exc:
            self._record_failure(exc, "JetBot connect failed")
            return False

        self._socket = sock
        self._set_status(
            connected=True,
            last_error="",
            last_connect_time=time.time(),
        )
        return True

    def _close_connection(self) -> None:
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
            self._record_failure(exc, "JetBot send failed")
            self._close_connection()
            return False

        self._set_status(last_send_time=time.time())
        return True

    def _send_stop_command(self) -> bool:
        return self._send_message({"type": "stop"})

    def _send_drive_command(self, command: DriveCommand) -> bool:
        return self._send_message(
            {
                "type": "drive",
                "left": command.left,
                "right": command.right,
                "client_time": time.time(),
            }
        )


class ArduinoBluetoothClient(DriveClientBase):
    def __init__(
        self,
        *,
        serial_port: str,
        serial_baud: int,
        max_pwm: int,
        drive_state: DriveState,
        send_rate_hz: float,
        verbose: bool,
    ) -> None:
        super().__init__(
            backend_name="arduino-bluetooth",
            transport_name="serial",
            endpoint=f"{serial_port} @ {serial_baud}",
            drive_state=drive_state,
            send_rate_hz=send_rate_hz,
            verbose=verbose,
        )
        self._serial_port = serial_port
        self._serial_baud = serial_baud
        self._max_pwm = max(1, min(int(max_pwm), 255))
        # This robot's current motor wiring interprets positive PWM as the
        # opposite of the laptop-side forward convention, so flip the sign here
        # once for every Arduino Bluetooth client.
        self._command_sign = -1
        self._serial: serial.Serial | None = None

    def _connect(self) -> bool:
        if self._serial is not None and self._serial.is_open:
            return True

        try:
            connection = serial.Serial(
                self._serial_port,
                baudrate=self._serial_baud,
                timeout=1.0,
                write_timeout=1.0,
            )
            time.sleep(0.15)
            connection.reset_input_buffer()
        except (serial.SerialException, OSError, ValueError) as exc:
            self._record_failure(exc, "Arduino Bluetooth connect failed")
            return False

        self._serial = connection
        self._set_status(
            connected=True,
            last_error="",
            last_connect_time=time.time(),
        )
        return True

    def _close_connection(self) -> None:
        connection = self._serial
        self._serial = None
        if connection is not None:
            try:
                connection.close()
            except (serial.SerialException, OSError):
                pass
        self._set_status(connected=False)

    def _send_line(self, line: str) -> bool:
        if self._serial is None and not self._connect():
            return False

        data = (line + "\n").encode("ascii", errors="ignore")
        try:
            assert self._serial is not None
            self._serial.write(data)
            self._serial.flush()
        except (serial.SerialException, OSError) as exc:
            self._record_failure(exc, "Arduino Bluetooth send failed")
            self._close_connection()
            return False

        self._set_status(last_send_time=time.time())
        return True

    def _pwm(self, value: float) -> int:
        return int(round(self._command_sign * clamp(value) * self._max_pwm))

    def _send_stop_command(self) -> bool:
        return self._send_line("S")

    def _send_drive_command(self, command: DriveCommand) -> bool:
        left_pwm = self._pwm(command.left)
        right_pwm = self._pwm(command.right)
        return self._send_line(f"D,{left_pwm},{right_pwm}")


def create_drive_client(
    *,
    drive_backend: str,
    drive_state: DriveState,
    send_rate_hz: float,
    verbose: bool,
    jetbot_host: str,
    jetbot_port: int,
    serial_port: str,
    serial_baud: int,
    serial_max_pwm: int,
) -> DriveClientBase:
    if drive_backend == "jetbot-socket":
        return JetBotSocketClient(
            host=jetbot_host,
            port=jetbot_port,
            drive_state=drive_state,
            send_rate_hz=send_rate_hz,
            verbose=verbose,
        )

    if drive_backend == "arduino-bluetooth":
        if not serial_port:
            raise ValueError("The arduino-bluetooth backend requires --serial-port, for example COM6.")
        return ArduinoBluetoothClient(
            serial_port=serial_port,
            serial_baud=serial_baud,
            max_pwm=serial_max_pwm,
            drive_state=drive_state,
            send_rate_hz=send_rate_hz,
            verbose=verbose,
        )

    raise ValueError(f"Unsupported drive backend: {drive_backend}")
