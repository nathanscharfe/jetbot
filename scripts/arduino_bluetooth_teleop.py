"""
Bluetooth teleoperation UI for the 4-wheel Arduino robot.

This script runs on the laptop, connects to the HC-05 Bluetooth serial COM
port, and streams left/right differential-drive commands to the Arduino using:

    D,<left_pwm>,<right_pwm>

The Arduino-side protocol is defined in:
    4 wheel arduino robot car/arduino_robot_controller/arduino_robot_controller.ino
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

from drive_backends import DriveState, create_drive_client


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teleoperate the Arduino Bluetooth robot from the laptop.")
    parser.add_argument("--serial-port", required=True, help="Bluetooth COM port for the HC-05, for example COM6.")
    parser.add_argument("--serial-baud", type=int, default=9600)
    parser.add_argument("--serial-max-pwm", type=int, default=255)
    parser.add_argument("--send-rate-hz", type=float, default=15.0)
    parser.add_argument("--speed", type=float, default=0.7, help="Normalized forward/reverse speed in [0, 1].")
    parser.add_argument("--turn-speed", type=float, default=0.5, help="Normalized turning speed in [0, 1].")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    drive_state = DriveState()
    drive_client = create_drive_client(
        drive_backend="arduino-bluetooth",
        drive_state=drive_state,
        send_rate_hz=args.send_rate_hz,
        verbose=args.verbose,
        jetbot_host="",
        jetbot_port=0,
        serial_port=args.serial_port,
        serial_baud=args.serial_baud,
        serial_max_pwm=args.serial_max_pwm,
    )
    drive_client.start()

    fig = plt.figure(figsize=(10, 6))
    status_ax = fig.add_axes([0.08, 0.56, 0.84, 0.32])
    status_ax.axis("off")

    fig.text(0.08, 0.92, "Arduino Bluetooth Teleop", fontsize=16, fontweight="bold")
    fig.text(0.08, 0.885, f"Endpoint: {args.serial_port} @ {args.serial_baud} baud", fontsize=10)
    fig.text(0.08, 0.49, "Drive Tuning", fontsize=11, fontweight="bold")
    fig.text(0.08, 0.46, "Speed", fontsize=9)
    fig.text(0.33, 0.46, "Turn Speed", fontsize=9)
    fig.text(0.50, 0.18, "Manual Drive (press and hold)", fontsize=11, fontweight="bold", ha="center")

    speed_box_ax = fig.add_axes([0.08, 0.37, 0.18, 0.07])
    turn_box_ax = fig.add_axes([0.33, 0.37, 0.18, 0.07])
    apply_button_ax = fig.add_axes([0.58, 0.37, 0.14, 0.07])

    speed_text = TextBox(speed_box_ax, "", initial=f"{args.speed:.2f}")
    turn_text = TextBox(turn_box_ax, "", initial=f"{args.turn_speed:.2f}")
    apply_button = Button(apply_button_ax, "Apply", color="#aed6f1", hovercolor="#d6eaf8")
    apply_button.label.set_fontsize(10)

    button_specs = [
        {"label": "Left", "rect": [0.15, 0.07, 0.12, 0.08], "color": "#9ec5fe"},
        {"label": "Forward", "rect": [0.30, 0.07, 0.12, 0.08], "color": "#8fd19e"},
        {"label": "Stop", "rect": [0.45, 0.07, 0.12, 0.08], "color": "#f5a3a3"},
        {"label": "Right", "rect": [0.60, 0.07, 0.12, 0.08], "color": "#9ec5fe"},
        {"label": "Reverse", "rect": [0.75, 0.07, 0.12, 0.08], "color": "#f7c97f"},
    ]

    settings = {
        "speed": args.speed,
        "turn_speed": args.turn_speed,
        "message": "Ready. Press and hold a button to drive.",
    }
    active_manual_label: str | None = None
    button_axes_to_label: dict[object, str] = {}

    for spec in button_specs:
        button_ax = fig.add_axes(spec["rect"])
        button = Button(button_ax, spec["label"], color=spec["color"], hovercolor="#dddddd")
        button.label.set_fontsize(10)
        button_axes_to_label[button_ax] = str(spec["label"])

    def apply_settings_from_boxes() -> bool:
        try:
            speed = float(speed_text.text)
            turn_speed = float(turn_text.text)
        except ValueError:
            settings["message"] = "Invalid entry. Use numeric speed values."
            return False

        if not (0.0 <= speed <= 1.0 and 0.0 <= turn_speed <= 1.0):
            settings["message"] = "Speed values must stay between 0.0 and 1.0."
            return False

        settings["speed"] = speed
        settings["turn_speed"] = turn_speed
        settings["message"] = f"Applied speed={speed:.2f}, turn_speed={turn_speed:.2f}."
        return True

    def command_for_label(label: str) -> tuple[float, float] | None:
        speed = settings["speed"]
        turn_speed = settings["turn_speed"]
        if label == "Forward":
            return speed, speed
        if label == "Reverse":
            return -speed, -speed
        if label == "Left":
            return -turn_speed, turn_speed
        if label == "Right":
            return turn_speed, -turn_speed
        return None

    def on_apply_clicked(_event) -> None:
        apply_settings_from_boxes()

    def on_mouse_press(event) -> None:
        nonlocal active_manual_label
        if event.inaxes not in button_axes_to_label:
            return

        label = button_axes_to_label[event.inaxes]
        if label == "Stop":
            active_manual_label = None
            drive_state.stop()
            settings["message"] = "Stop requested."
            return

        command = command_for_label(label)
        if command is None:
            return

        active_manual_label = label
        drive_state.set(*command)
        settings["message"] = f"Manual {label.lower()} command active."

    def on_mouse_release(_event) -> None:
        nonlocal active_manual_label
        if active_manual_label is None:
            return

        active_manual_label = None
        drive_state.stop()
        settings["message"] = "Manual control released."

    def on_key_press(event) -> None:
        nonlocal active_manual_label
        if event.key is None:
            return

        key = event.key.lower()
        key_to_label = {
            "up": "Forward",
            "w": "Forward",
            "down": "Reverse",
            "s": "Reverse",
            "left": "Left",
            "a": "Left",
            "right": "Right",
            "d": "Right",
            " ": "Stop",
            "escape": "Stop",
            "x": "Stop",
        }
        label = key_to_label.get(key)
        if label is None:
            return
        if label == "Stop":
            active_manual_label = None
            drive_state.stop()
            settings["message"] = "Stop requested from keyboard."
            return

        command = command_for_label(label)
        if command is None:
            return
        active_manual_label = label
        drive_state.set(*command)
        settings["message"] = f"Manual {label.lower()} command active from keyboard."

    def on_key_release(event) -> None:
        nonlocal active_manual_label
        if event.key is None:
            return

        if active_manual_label is None:
            return

        key = event.key.lower()
        if key in {"up", "w", "down", "s", "left", "a", "right", "d"}:
            active_manual_label = None
            drive_state.stop()
            settings["message"] = "Keyboard control released."

    apply_button.on_clicked(on_apply_clicked)
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    def update_status(_frame_index: int) -> None:
        drive_command = drive_state.snapshot()
        client_status = drive_client.snapshot()

        status_lines = [
            f"Drive backend: {client_status['backend']}",
            f"Transport: {client_status['transport']}",
            f"Endpoint: {client_status['endpoint']}",
            f"Control link: {'connected' if client_status['connected'] else 'disconnected'}",
            f"Drive command: ({drive_command.left:.2f}, {drive_command.right:.2f})",
            f"Speed: {settings['speed']:.2f}",
            f"Turn speed: {settings['turn_speed']:.2f}",
            f"PWM range: +/-{args.serial_max_pwm}",
            f"Send rate (Hz): {args.send_rate_hz:.1f}",
            f"Message: {settings['message']}",
        ]
        if client_status["last_error"]:
            status_lines.append(f"Control error: {client_status['last_error']}")

        status_ax.cla()
        status_ax.axis("off")
        status_ax.text(
            0.0,
            1.0,
            "\n".join(status_lines),
            transform=status_ax.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "lightgray"},
        )

    animation = FuncAnimation(fig, update_status, interval=100, cache_frame_data=False)

    try:
        plt.show()
    finally:
        drive_client.stop()
        drive_client.join(timeout=1.0)
        del animation


if __name__ == "__main__":
    main()
