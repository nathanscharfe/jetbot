"""
Vicon room controller with MPPI go-to-target control for JetBot.

Run this on the laptop while `notebooks/jetbot_socket_server.ipynb` is running on
the JetBot. The script:
- receives the Vicon UDP Object Stream
- shows the room and the robot pose in 3D
- lets the user enter a target X/Y coordinate and click Go
- uses MPPI to continuously choose differential-drive commands toward the target
- stops automatically once the robot gets within an epsilon of the target

This version does not include obstacle costs or path-following logic yet. It is
an MPPI point-to-point controller for the same basic go-to task handled by the
continuous heuristic controller.
"""

from __future__ import annotations

import argparse
import math
import random
import threading
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, TextBox

from vicon_teleop_viewer import (
    DEFAULT_OBJECT_NAME,
    DEFAULT_VICON_SOURCE_IP,
    DriveState,
    TeleopClient,
    clamp,
)
from vicon_udp_viewer import (
    SharedState,
    ViconReceiver,
    build_status_lines,
    draw_pose_marker,
    draw_room,
    rotate_vector,
    rotation_matrix_xyz,
    set_axes_equal,
)


DEFAULT_JETBOT_HOST = "192.168.0.86"
DEFAULT_JETBOT_PORT = 8765
UNSET = object()


def wrap_to_pi(angle_radians: float) -> float:
    return (angle_radians + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class TargetState:
    active: bool = False
    reached: bool = False
    target_x: float = 0.0
    target_y: float = 0.0
    last_distance: float | None = None
    last_heading_error_deg: float | None = None
    mode: str = "idle"
    message: str = "Enter a target and click Go."


@dataclass
class MPPITuning:
    epsilon: float = 150.0
    heading_offset_deg: float = 0.0


class MPPIController(threading.Thread):
    def __init__(
        self,
        shared_state: SharedState,
        selected_name: str,
        drive_state: DriveState,
        speed: float,
        turn_speed: float,
        min_forward_speed: float,
        epsilon: float,
        slow_radius: float,
        heading_gain: float,
        control_rate_hz: float,
        heading_offset_deg: float,
        horizon_steps: int,
        num_samples: int,
        temperature: float,
        noise_std: float,
        model_linear_speed: float,
        model_angular_speed: float,
        heuristic_blend: float,
        distance_cost_weight: float,
        heading_cost_weight: float,
        terminal_distance_cost_weight: float,
        terminal_heading_cost_weight: float,
        control_effort_weight: float,
        control_slew_weight: float,
        reverse_cost_weight: float,
    ) -> None:
        super().__init__(daemon=True)
        self._shared_state = shared_state
        self._selected_name = selected_name
        self._drive_state = drive_state
        self._speed = speed
        self._turn_speed = turn_speed
        self._min_forward_speed = min_forward_speed
        self._slow_radius = slow_radius
        self._heading_gain = heading_gain
        self._period = 1.0 / max(control_rate_hz, 1.0)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._state = TargetState()
        self._tuning = MPPITuning(
            epsilon=epsilon,
            heading_offset_deg=heading_offset_deg,
        )

        self._horizon_steps = max(horizon_steps, 1)
        self._num_samples = max(num_samples, 1)
        self._temperature = max(temperature, 1e-6)
        self._noise_std = max(noise_std, 1e-6)
        self._model_linear_speed = model_linear_speed
        self._model_angular_speed = model_angular_speed
        self._heuristic_blend = clamp(heuristic_blend, 0.0, 1.0)

        self._distance_scale = max(self._slow_radius, 1.0)
        self._distance_cost_weight = distance_cost_weight
        self._heading_cost_weight = heading_cost_weight
        self._terminal_distance_cost_weight = terminal_distance_cost_weight
        self._terminal_heading_cost_weight = terminal_heading_cost_weight
        self._control_effort_weight = control_effort_weight
        self._control_slew_weight = control_slew_weight
        self._reverse_cost_weight = reverse_cost_weight

        self._nominal_controls = [(0.0, 0.0)] * self._horizon_steps
        self._rng = random.Random()

    def set_target(self, target_x: float, target_y: float) -> None:
        with self._lock:
            self._state = TargetState(
                active=True,
                reached=False,
                target_x=target_x,
                target_y=target_y,
                last_distance=None,
                last_heading_error_deg=None,
                mode="target_set",
                message="Target accepted. Running MPPI rollouts toward goal.",
            )
            self._nominal_controls = [(0.0, 0.0)] * self._horizon_steps

    def clear_target(self, message: str = "Target cleared.") -> None:
        self._drive_state.stop()
        with self._lock:
            self._state.active = False
            self._state.mode = "idle"
            self._state.message = message
            self._nominal_controls = [(0.0, 0.0)] * self._horizon_steps

    def stop(self) -> None:
        self._stop_event.set()
        self.clear_target("Controller stopped.")

    def snapshot(self) -> TargetState:
        with self._lock:
            return TargetState(
                active=self._state.active,
                reached=self._state.reached,
                target_x=self._state.target_x,
                target_y=self._state.target_y,
                last_distance=self._state.last_distance,
                last_heading_error_deg=self._state.last_heading_error_deg,
                mode=self._state.mode,
                message=self._state.message,
            )

    def tuning_snapshot(self) -> MPPITuning:
        with self._lock:
            return MPPITuning(
                epsilon=self._tuning.epsilon,
                heading_offset_deg=self._tuning.heading_offset_deg,
            )

    def set_message(self, message: str) -> None:
        self._update_state(message=message)

    def update_tuning(
        self,
        *,
        epsilon: float,
        heading_offset_deg: float,
    ) -> tuple[bool, str]:
        if epsilon <= 0.0:
            return False, "Epsilon must be positive."

        with self._lock:
            self._tuning.epsilon = epsilon
            self._state.message = f"Tuning updated: epsilon={epsilon:.1f}, angle correction={heading_offset_deg:.1f} deg"
            self._tuning.heading_offset_deg = heading_offset_deg

        return True, "Tuning updated."

    def _update_state(
        self,
        *,
        active: bool | None = None,
        reached: bool | None = None,
        last_distance: float | None | object = UNSET,
        last_heading_error_deg: float | None | object = UNSET,
        mode: str | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            if active is not None:
                self._state.active = active
            if reached is not None:
                self._state.reached = reached
            if last_distance is not UNSET:
                self._state.last_distance = last_distance
            if last_heading_error_deg is not UNSET:
                self._state.last_heading_error_deg = last_heading_error_deg
            if mode is not None:
                self._state.mode = mode
            if message is not None:
                self._state.message = message

    def _current_target(self) -> tuple[bool, float, float]:
        with self._lock:
            return self._state.active, self._state.target_x, self._state.target_y

    def _heuristic_command(
        self,
        x: float,
        y: float,
        heading: float,
        target_x: float,
        target_y: float,
    ) -> tuple[float, float]:
        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)

        if distance >= self._slow_radius:
            base_forward = self._speed
        else:
            blend = max(distance / max(self._slow_radius, 1e-6), 0.0)
            base_forward = self._min_forward_speed + (self._speed - self._min_forward_speed) * blend
            base_forward = min(base_forward, self._speed)

        target_heading = math.atan2(dy, dx)
        heading_error = wrap_to_pi(target_heading - heading)
        heading_alignment = max(0.0, math.cos(heading_error))
        forward = base_forward * (heading_alignment**2)
        steering = clamp(
            self._heading_gain * heading_error,
            -self._turn_speed,
            self._turn_speed,
        )
        return clamp(forward - steering), clamp(forward + steering)

    def _simulate_step(
        self,
        x: float,
        y: float,
        heading: float,
        left: float,
        right: float,
    ) -> tuple[float, float, float]:
        forward_command = 0.5 * (left + right)
        turn_command = 0.5 * (right - left)

        linear_speed = self._model_linear_speed * forward_command
        angular_speed = self._model_angular_speed * turn_command

        x_next = x + linear_speed * math.cos(heading) * self._period
        y_next = y + linear_speed * math.sin(heading) * self._period
        heading_next = wrap_to_pi(heading + angular_speed * self._period)
        return x_next, y_next, heading_next

    def _sequence_cost(
        self,
        controls: list[tuple[float, float]],
        x: float,
        y: float,
        heading: float,
        target_x: float,
        target_y: float,
        current_command: tuple[float, float],
    ) -> float:
        total_cost = 0.0
        prev_left, prev_right = current_command

        for left, right in controls:
            x, y, heading = self._simulate_step(x, y, heading, left, right)

            dx = target_x - x
            dy = target_y - y
            distance = math.hypot(dx, dy)
            target_heading = math.atan2(dy, dx)
            heading_error = wrap_to_pi(target_heading - heading)
            forward_command = 0.5 * (left + right)

            distance_term = distance / self._distance_scale
            total_cost += self._distance_cost_weight * (distance_term**2)
            total_cost += self._heading_cost_weight * (heading_error**2)
            total_cost += self._control_effort_weight * (left * left + right * right)
            total_cost += self._control_slew_weight * (
                (left - prev_left) * (left - prev_left) + (right - prev_right) * (right - prev_right)
            )
            total_cost += self._reverse_cost_weight * max(-forward_command, 0.0) ** 2

            prev_left, prev_right = left, right

        terminal_dx = target_x - x
        terminal_dy = target_y - y
        terminal_distance = math.hypot(terminal_dx, terminal_dy) / self._distance_scale
        terminal_heading_error = wrap_to_pi(math.atan2(terminal_dy, terminal_dx) - heading)

        total_cost += self._terminal_distance_cost_weight * (terminal_distance**2)
        total_cost += self._terminal_heading_cost_weight * (terminal_heading_error**2)
        return total_cost

    def _build_heuristic_sequence(
        self,
        x: float,
        y: float,
        heading: float,
        target_x: float,
        target_y: float,
    ) -> list[tuple[float, float]]:
        sequence: list[tuple[float, float]] = []

        for _ in range(self._horizon_steps):
            command = self._heuristic_command(x, y, heading, target_x, target_y)
            sequence.append(command)
            x, y, heading = self._simulate_step(x, y, heading, *command)

        return sequence

    def _mppi_command(
        self,
        x: float,
        y: float,
        heading: float,
        target_x: float,
        target_y: float,
        current_command: tuple[float, float],
    ) -> tuple[float, float]:
        heuristic_sequence = self._build_heuristic_sequence(x, y, heading, target_x, target_y)
        shifted_sequence = self._nominal_controls[1:] + [self._nominal_controls[-1]]

        nominal_sequence: list[tuple[float, float]] = []
        for shifted, heuristic in zip(shifted_sequence, heuristic_sequence):
            left = (1.0 - self._heuristic_blend) * shifted[0] + self._heuristic_blend * heuristic[0]
            right = (1.0 - self._heuristic_blend) * shifted[1] + self._heuristic_blend * heuristic[1]
            nominal_sequence.append((clamp(left), clamp(right)))

        rollout_costs: list[float] = []
        rollout_noises: list[list[tuple[float, float]]] = []

        for _ in range(self._num_samples):
            candidate_controls: list[tuple[float, float]] = []
            candidate_noise: list[tuple[float, float]] = []

            for left_nominal, right_nominal in nominal_sequence:
                noise_left = self._rng.gauss(0.0, self._noise_std)
                noise_right = self._rng.gauss(0.0, self._noise_std)
                candidate_noise.append((noise_left, noise_right))
                candidate_controls.append(
                    (
                        clamp(left_nominal + noise_left),
                        clamp(right_nominal + noise_right),
                    )
                )

            cost = self._sequence_cost(
                candidate_controls,
                x,
                y,
                heading,
                target_x,
                target_y,
                current_command,
            )
            rollout_costs.append(cost)
            rollout_noises.append(candidate_noise)

        min_cost = min(rollout_costs)
        weights = [math.exp(-(cost - min_cost) / self._temperature) for cost in rollout_costs]
        weight_sum = sum(weights)
        if weight_sum <= 1e-12:
            self._nominal_controls = nominal_sequence
            return nominal_sequence[0]

        updated_sequence: list[tuple[float, float]] = []
        for step_index, (left_nominal, right_nominal) in enumerate(nominal_sequence):
            delta_left = 0.0
            delta_right = 0.0
            for weight, noise_sequence in zip(weights, rollout_noises):
                noise_left, noise_right = noise_sequence[step_index]
                delta_left += weight * noise_left
                delta_right += weight * noise_right

            delta_left /= weight_sum
            delta_right /= weight_sum
            updated_sequence.append(
                (
                    clamp(left_nominal + delta_left),
                    clamp(right_nominal + delta_right),
                )
            )

        self._nominal_controls = updated_sequence
        return updated_sequence[0]

    def run(self) -> None:
        while not self._stop_event.is_set():
            active, target_x, target_y = self._current_target()
            tuning = self.tuning_snapshot()

            if not active:
                self._stop_event.wait(self._period)
                continue

            snapshot = self._shared_state.snapshot(self._selected_name)
            visible = snapshot["visible"]
            if self._selected_name not in visible:
                self._drive_state.stop()
                self._update_state(
                    last_distance=None,
                    last_heading_error_deg=None,
                    mode="waiting_for_pose",
                    message="Waiting for a fresh pose for the selected object.",
                )
                self._stop_event.wait(self._period)
                continue

            pose = visible[self._selected_name]["pose"]
            dx = target_x - pose.tx
            dy = target_y - pose.ty
            distance = math.hypot(dx, dy)

            rotation = rotation_matrix_xyz(pose.rx, pose.ry, pose.rz)
            physical_forward = rotate_vector(rotation, (0.0, 1.0, 0.0))
            current_heading = math.atan2(physical_forward[1], physical_forward[0]) + math.radians(
                tuning.heading_offset_deg
            )
            current_heading = wrap_to_pi(current_heading)
            target_heading = math.atan2(dy, dx)
            heading_error = wrap_to_pi(target_heading - current_heading)
            heading_error_deg = math.degrees(heading_error)

            if distance <= tuning.epsilon:
                self._drive_state.stop()
                self._update_state(
                    active=False,
                    reached=True,
                    last_distance=distance,
                    last_heading_error_deg=heading_error_deg,
                    mode="arrived",
                    message=f"Reached target within epsilon ({tuning.epsilon:.1f}).",
                )
                self._nominal_controls = [(0.0, 0.0)] * self._horizon_steps
                self._stop_event.wait(self._period)
                continue

            drive_command = self._drive_state.snapshot()
            left, right = self._mppi_command(
                pose.tx,
                pose.ty,
                current_heading,
                target_x,
                target_y,
                (drive_command.left, drive_command.right),
            )

            self._drive_state.set(left, right)
            self._update_state(
                last_distance=distance,
                last_heading_error_deg=heading_error_deg,
                mode="mppi",
                message="Driving toward the target with MPPI.",
            )
            self._stop_event.wait(self._period)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive the Vicon UDP stream, show the room in 3D, and drive the JetBot to a target with an MPPI controller."
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
    parser.add_argument("--speed", type=float, default=0.7)
    parser.add_argument("--turn-speed", type=float, default=0.5)
    parser.add_argument("--min-forward-speed", type=float, default=0.22)
    parser.add_argument("--epsilon", type=float, default=150.0, help="Target tolerance in Vicon translation units.")
    parser.add_argument("--slow-radius", type=float, default=700.0, help="Start slowing down within this radius.")
    parser.add_argument("--heading-gain", type=float, default=0.9)
    parser.add_argument("--forward-offset-deg", type=float, default=0.0)
    parser.add_argument("--control-rate-hz", type=float, default=15.0)
    parser.add_argument("--jetbot-host", default=DEFAULT_JETBOT_HOST)
    parser.add_argument("--jetbot-port", type=int, default=DEFAULT_JETBOT_PORT)
    parser.add_argument("--send-rate-hz", type=float, default=20.0)
    parser.add_argument("--horizon-steps", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.18)
    parser.add_argument("--model-linear-speed", type=float, default=700.0)
    parser.add_argument("--model-angular-speed", type=float, default=3.0)
    parser.add_argument("--heuristic-blend", type=float, default=0.35)
    parser.add_argument("--distance-cost-weight", type=float, default=1.0)
    parser.add_argument("--heading-cost-weight", type=float, default=0.35)
    parser.add_argument("--terminal-distance-cost-weight", type=float, default=18.0)
    parser.add_argument("--terminal-heading-cost-weight", type=float, default=1.0)
    parser.add_argument("--control-effort-weight", type=float, default=0.03)
    parser.add_argument("--control-slew-weight", type=float, default=0.08)
    parser.add_argument("--reverse-cost-weight", type=float, default=0.25)
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
    input_to_vicon_scale = 1.0 if args.units == "mm" else 1000.0
    room_floor_z = room_center[2] - room_size[2] / 2.0

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

    controller = MPPIController(
        shared_state=shared_state,
        selected_name=selected_object_name or args.object_name,
        drive_state=drive_state,
        speed=args.speed,
        turn_speed=args.turn_speed,
        min_forward_speed=args.min_forward_speed,
        epsilon=args.epsilon,
        slow_radius=args.slow_radius,
        heading_gain=args.heading_gain,
        control_rate_hz=args.control_rate_hz,
        heading_offset_deg=args.forward_offset_deg,
        horizon_steps=args.horizon_steps,
        num_samples=args.num_samples,
        temperature=args.temperature,
        noise_std=args.noise_std,
        model_linear_speed=args.model_linear_speed,
        model_angular_speed=args.model_angular_speed,
        heuristic_blend=args.heuristic_blend,
        distance_cost_weight=args.distance_cost_weight,
        heading_cost_weight=args.heading_cost_weight,
        terminal_distance_cost_weight=args.terminal_distance_cost_weight,
        terminal_heading_cost_weight=args.terminal_heading_cost_weight,
        control_effort_weight=args.control_effort_weight,
        control_slew_weight=args.control_slew_weight,
        reverse_cost_weight=args.reverse_cost_weight,
    )

    receiver.start()
    teleop_client.start()
    controller.start()

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_axes([0.26, 0.36, 0.56, 0.56], projection="3d")
    status_ax = fig.add_axes([0.03, 0.39, 0.20, 0.51])
    status_ax.axis("off")

    fig.text(0.08, 0.315, "MPPI Go-To Controller", fontsize=11, fontweight="bold")
    fig.text(0.08, 0.292, f"Target X ({args.units})", fontsize=9)
    fig.text(0.26, 0.292, f"Target Y ({args.units})", fontsize=9)
    fig.text(0.08, 0.205, "Controller Tuning", fontsize=11, fontweight="bold")
    fig.text(0.08, 0.182, f"Epsilon ({args.units})", fontsize=9)
    fig.text(0.24, 0.182, "Angle Correction (deg)", fontsize=9)
    fig.text(0.50, 0.095, "Manual Drive (press and hold)", fontsize=11, fontweight="bold", ha="center")

    x_box_ax = fig.add_axes([0.08, 0.235, 0.14, 0.055])
    y_box_ax = fig.add_axes([0.26, 0.235, 0.14, 0.055])
    go_button_ax = fig.add_axes([0.45, 0.235, 0.11, 0.055])
    stop_button_ax = fig.add_axes([0.60, 0.235, 0.16, 0.055])
    epsilon_box_ax = fig.add_axes([0.08, 0.125, 0.12, 0.055])
    offset_box_ax = fig.add_axes([0.24, 0.125, 0.12, 0.055])
    apply_button_ax = fig.add_axes([0.40, 0.125, 0.11, 0.055])

    x_text = TextBox(x_box_ax, "", initial="0")
    y_text = TextBox(y_box_ax, "", initial="0")
    epsilon_text = TextBox(epsilon_box_ax, "", initial=f"{args.epsilon * display_scale:.1f}")
    offset_text = TextBox(offset_box_ax, "", initial=f"{args.forward_offset_deg:.1f}")
    go_button = Button(go_button_ax, "Go", color="#8fd19e", hovercolor="#d5f5e3")
    stop_button = Button(stop_button_ax, "Stop Go-To", color="#f5a3a3", hovercolor="#fadbd8")
    apply_button = Button(apply_button_ax, "Apply", color="#aed6f1", hovercolor="#d6eaf8")
    go_button.label.set_fontsize(10)
    stop_button.label.set_fontsize(10)
    apply_button.label.set_fontsize(10)

    color_cycle = list(plt.get_cmap("tab10").colors)
    active_manual_command: tuple[float, float] | None = None

    def apply_tuning_from_boxes() -> bool:
        try:
            epsilon_display = float(epsilon_text.text)
            heading_offset_deg = float(offset_text.text)
        except ValueError:
            controller.set_message("Invalid tuning entry. Use numeric values for epsilon and angle correction.")
            return False

        ok, message = controller.update_tuning(
            epsilon=epsilon_display * input_to_vicon_scale,
            heading_offset_deg=heading_offset_deg,
        )
        controller.set_message(message)
        return ok

    def on_go_clicked(_event) -> None:
        if not apply_tuning_from_boxes():
            return
        try:
            target_x = float(x_text.text) * input_to_vicon_scale
            target_y = float(y_text.text) * input_to_vicon_scale
        except ValueError:
            controller.clear_target("Invalid target entry. Enter numeric X and Y.")
            return

        controller.set_target(target_x, target_y)

    def on_stop_clicked(_event) -> None:
        controller.clear_target("Stopped by user.")

    manual_button_specs = [
        {
            "label": "Left",
            "rect": [0.18, 0.03, 0.11, 0.055],
            "command": (-args.turn_speed, args.turn_speed),
            "color": "#9ec5fe",
        },
        {
            "label": "Forward",
            "rect": [0.31, 0.03, 0.11, 0.055],
            "command": (args.speed, args.speed),
            "color": "#8fd19e",
        },
        {
            "label": "Stop",
            "rect": [0.44, 0.03, 0.11, 0.055],
            "command": None,
            "color": "#f5a3a3",
        },
        {
            "label": "Right",
            "rect": [0.57, 0.03, 0.11, 0.055],
            "command": (args.turn_speed, -args.turn_speed),
            "color": "#9ec5fe",
        },
        {
            "label": "Reverse",
            "rect": [0.70, 0.03, 0.11, 0.055],
            "command": (-args.speed, -args.speed),
            "color": "#f7c97f",
        },
    ]
    manual_axes_to_spec: dict[object, dict[str, object]] = {}
    manual_button_widgets: list[Button] = []

    for spec in manual_button_specs:
        button_ax = fig.add_axes(spec["rect"])
        button = Button(button_ax, spec["label"], color=spec["color"], hovercolor="#dddddd")
        button.label.set_fontsize(10)
        manual_axes_to_spec[button_ax] = spec
        manual_button_widgets.append(button)

    def on_apply_clicked(_event) -> None:
        apply_tuning_from_boxes()

    def on_mouse_press(event) -> None:
        nonlocal active_manual_command
        if event.inaxes not in manual_axes_to_spec:
            return

        spec = manual_axes_to_spec[event.inaxes]
        label = str(spec["label"])
        command = spec["command"]

        if command is None:
            active_manual_command = None
            controller.clear_target("Manual stop requested.")
            return

        active_manual_command = command
        controller.clear_target(f"Manual {label.lower()} command active.")
        drive_state.set(*command)

    def on_mouse_release(_event) -> None:
        nonlocal active_manual_command
        if active_manual_command is None:
            return
        active_manual_command = None
        drive_state.stop()
        controller.set_message("Manual control released.")

    go_button.on_clicked(on_go_clicked)
    stop_button.on_clicked(on_stop_clicked)
    apply_button.on_clicked(on_apply_clicked)
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)

    def update_plot(_frame_index: int) -> None:
        snapshot = shared_state.snapshot(selected_object_name)
        visible = snapshot["visible"]
        drive_command = drive_state.snapshot()
        teleop_status = teleop_client.snapshot()
        target_state = controller.snapshot()
        tuning_state = controller.tuning_snapshot()

        ax.cla()
        ax.set_xlabel(f"X ({args.units})")
        ax.set_ylabel(f"Y ({args.units})")
        ax.set_zlabel(f"Z ({args.units})")
        ax.set_title("Vicon MPPI Go-To Controller")
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
                    heading_correction_deg=tuning_state.heading_offset_deg,
                )
            )

        if target_state.active or target_state.reached:
            target_x_display = target_state.target_x * display_scale
            target_y_display = target_state.target_y * display_scale
            ax.scatter(
                [target_x_display],
                [target_y_display],
                [room_floor_z],
                marker="*",
                s=160,
                color="#8e44ad",
                edgecolor="white",
                linewidths=0.8,
                zorder=6,
            )
            ax.text(target_x_display, target_y_display, room_floor_z, " target", color="#6c3483")
            plotted_points.append((target_x_display, target_y_display, room_floor_z))
            legend_handles.append(
                Line2D([0], [0], color="#8e44ad", marker="*", linestyle="None", markersize=10, label="target")
            )

            if selected_object_name in visible:
                pose = visible[selected_object_name]["pose"]
                ax.plot(
                    [pose.tx * display_scale, target_x_display],
                    [pose.ty * display_scale, target_y_display],
                    [pose.tz * display_scale, room_floor_z],
                    color="#8e44ad",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.5,
                )

        ax.scatter([0.0], [0.0], [0.0], color="black", marker="x", s=40)
        set_axes_equal(ax, plotted_points, padding=args.axis_padding * display_scale)

        if legend_handles and len(legend_handles) <= 10:
            deduped_handles: dict[str, Line2D] = {}
            for handle in legend_handles:
                deduped_handles[handle.get_label()] = handle
            ax.legend(handles=list(deduped_handles.values()), loc="upper left", bbox_to_anchor=(1.02, 1.0))

        status_lines = build_status_lines(
            snapshot,
            selected_object_name,
            args.units,
            display_scale,
        )
        if snapshot["last_packet_time"] is None:
            status_lines.append("Waiting for UDP packets...")
        elif selected_object_name and selected_object_name not in visible:
            status_lines.append("Receiving packets, but the selected object is not visible.")

        target_x_status = target_state.target_x * display_scale
        target_y_status = target_state.target_y * display_scale
        status_lines.extend(
            [
                "",
                f"JetBot server: {teleop_status['host']}:{teleop_status['port']}",
                f"Control link: {'connected' if teleop_status['connected'] else 'disconnected'}",
                f"Drive command: ({drive_command.left:.2f}, {drive_command.right:.2f})",
                f"Controller mode: {target_state.mode}",
                f"Target ({args.units}): ({target_x_status:.3f}, {target_y_status:.3f})",
                f"Epsilon ({args.units}): {tuning_state.epsilon * display_scale:.3f}",
                f"Angle correction (deg): {tuning_state.heading_offset_deg:.1f}",
                f"MPPI: samples={args.num_samples}, horizon={args.horizon_steps}, temp={args.temperature:.2f}",
                f"Model: v={args.model_linear_speed:.1f} {args.units}/s, w={args.model_angular_speed:.2f} rad/s",
            ]
        )

        if target_state.last_distance is not None:
            status_lines.append(f"Distance to target ({args.units}): {target_state.last_distance * display_scale:.3f}")
        if target_state.last_heading_error_deg is not None:
            status_lines.append(f"Heading error (deg): {target_state.last_heading_error_deg:.1f}")

        status_lines.extend(
            [
                f"Message: {target_state.message}",
                "Target entry uses the same units shown on the axes.",
            ]
        )

        if teleop_status["last_error"]:
            status_lines.append(f"Control error: {teleop_status['last_error']}")

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
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "lightgray"},
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
        controller.stop()
        teleop_client.stop()
        receiver.stop()
        controller.join(timeout=1.0)
        teleop_client.join(timeout=1.0)
        receiver.join(timeout=1.0)
        del animation


if __name__ == "__main__":
    main()
