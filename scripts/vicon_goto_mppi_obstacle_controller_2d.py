"""
2D driving-plane version of the obstacle-aware MPPI go-to controller.

This runs the same obstacle-aware MPPI control loop as
scripts/vicon_goto_mppi_obstacle_controller.py, but renders only the XY driving
plane instead of a full 3D room view.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from drive_backends import DriveState, create_drive_client
from mppi_startup_ui import show_mppi_startup_config_tk
from vicon_goto_mppi_controller_2d import draw_pose_marker_2d, draw_room_2d, set_axes_equal_2d
from vicon_goto_mppi_obstacle_controller import (
    ObstacleAwareMPPIController,
    collect_obstacles,
    make_parser as make_base_parser,
)
from vicon_udp_viewer import SharedState, ViconReceiver, build_status_lines

DEFAULTS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "vicon_goto_mppi_obstacle_controller_2d_defaults.json"


def make_parser():
    parser = make_base_parser()
    parser.description = (
        "Receive the Vicon UDP stream, show the XY driving plane, and drive the "
        "selected robot to a target with an obstacle-aware MPPI controller."
    )
    return parser


def _load_saved_defaults(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        parsed = json.loads(data)
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _explicit_cli_destinations(parser, argv: list[str]) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option_string in action.option_strings:
            option_to_dest[option_string] = action.dest

    explicit_dests: set[str] = set()
    for token in argv:
        if token == "--":
            break
        option = token.split("=", 1)[0] if token.startswith("--") else token
        dest = option_to_dest.get(option)
        if dest:
            explicit_dests.add(dest)
    return explicit_dests


def draw_obstacle_circle_2d(
    ax,
    *,
    x: float,
    y: float,
    radius: float,
    color: str,
) -> list[tuple[float, float]]:
    if radius <= 0.0:
        return []

    samples = 48
    points = [
        (
            x + radius * math.cos(2.0 * math.pi * index / samples),
            y + radius * math.sin(2.0 * math.pi * index / samples),
        )
        for index in range(samples + 1)
    ]
    ax.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=color,
        linestyle=":",
        linewidth=1.3,
        alpha=0.85,
    )
    return points


class ObstacleController2DApp:
    def __init__(
        self,
        *,
        args,
        shared_state: SharedState,
        receiver: ViconReceiver,
        drive_state: DriveState,
        teleop_client,
        controller: ObstacleAwareMPPIController,
        selected_object_name: str | None,
        display_scale: float,
        room_center_xy: tuple[float, float],
        room_size_xy: tuple[float, float],
        robot_footprint: float,
        robot_axis_length: float,
        input_to_vicon_scale: float,
    ) -> None:
        self._args = args
        self._shared_state = shared_state
        self._receiver = receiver
        self._drive_state = drive_state
        self._teleop_client = teleop_client
        self._controller = controller
        self._selected_object_name = selected_object_name
        self._display_scale = display_scale
        self._room_center_xy = room_center_xy
        self._room_size_xy = room_size_xy
        self._robot_footprint = robot_footprint
        self._robot_axis_length = robot_axis_length
        self._input_to_vicon_scale = input_to_vicon_scale
        self._closed = False
        self._shutdown_complete = False
        self._restart_requested = False
        self._update_job: str | None = None
        self._active_manual_command: tuple[float, float] | None = None

        self._root = tk.Tk()
        self._root.title("Vicon MPPI Obstacle Controller (2D)")
        self._root.geometry("1600x960")
        self._root.minsize(1280, 820)
        try:
            self._root.state("zoomed")
        except tk.TclError:
            pass
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.bind("<Escape>", lambda _event: self._on_close())
        self._root.bind_all("<ButtonRelease-1>", self._on_global_button_release, add="+")

        self._target_x_var = tk.StringVar(value="0")
        self._target_y_var = tk.StringVar(value="0")

        self._build_ui()
        self._schedule_refresh()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self._root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(1, weight=1)
        outer.grid_rowconfigure(0, weight=1)

        sidebar = ttk.Frame(outer)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        plot_panel = ttk.Frame(outer)
        plot_panel.grid(row=0, column=1, sticky="nsew")
        plot_panel.grid_rowconfigure(0, weight=1)
        plot_panel.grid_columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="MPPI Obstacle 2D", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            sidebar,
            text="Tk controls on the left, 2D Vicon driving plane on the right.",
            foreground="#555555",
        ).pack(anchor="w", pady=(2, 10))

        nav_frame = ttk.Frame(sidebar)
        nav_frame.pack(fill="x", pady=(0, 10))
        ttk.Button(nav_frame, text="Back to Setup", command=self._on_back_to_setup).pack(fill="x")

        self._build_target_controls(sidebar)
        self._build_view_controls(sidebar)
        self._build_manual_controls(sidebar)
        self._build_status_panel(sidebar)

        self._figure = Figure(figsize=(8.5, 8.0), dpi=100)
        self._figure.subplots_adjust(left=0.08, right=0.80, bottom=0.09, top=0.92)
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._figure, master=plot_panel)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_target_controls(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Go-To Target", padding=8)
        frame.pack(fill="x", pady=(0, 10))
        for column_index in range(2):
            frame.grid_columnconfigure(column_index, weight=1)

        ttk.Label(frame, text=f"Target X ({self._args.units})").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text=f"Target Y ({self._args.units})").grid(row=0, column=1, sticky="w")
        x_entry = ttk.Entry(frame, textvariable=self._target_x_var, width=14)
        y_entry = ttk.Entry(frame, textvariable=self._target_y_var, width=14)
        x_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(2, 8))
        y_entry.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(2, 8))
        x_entry.bind("<Return>", lambda _event: self._on_go_clicked())
        y_entry.bind("<Return>", lambda _event: self._on_go_clicked())

        button_row = ttk.Frame(frame)
        button_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        button_row.grid_columnconfigure(0, weight=1)
        button_row.grid_columnconfigure(1, weight=1)

        tk.Button(
            button_row,
            text="Go",
            bg="#8fd19e",
            activebackground="#d5f5e3",
            command=self._on_go_clicked,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Button(
            button_row,
            text="Stop Go-To",
            bg="#f5a3a3",
            activebackground="#fadbd8",
            command=self._on_stop_clicked,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def _build_view_controls(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Display", padding=8)
        frame.pack(fill="x", pady=(0, 10))
        ttk.Label(
            frame,
            text="Rollout paths are always shown on the graph.",
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Manual drive cancels the current go-to target.",
            foreground="#555555",
        ).pack(anchor="w", pady=(6, 0))

    def _build_manual_controls(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Manual Drive (press and hold)", padding=8)
        frame.pack(fill="x", pady=(0, 10))
        for column_index in range(3):
            frame.grid_columnconfigure(column_index, weight=1)

        self._create_hold_button(
            frame,
            text="Forward",
            row=0,
            column=1,
            command=(self._args.speed, self._args.speed),
            color="#8fd19e",
        )
        self._create_hold_button(
            frame,
            text="Left",
            row=1,
            column=0,
            command=(-self._args.turn_speed, self._args.turn_speed),
            color="#9ec5fe",
        )
        tk.Button(
            frame,
            text="Stop",
            bg="#f5a3a3",
            activebackground="#fadbd8",
            command=self._on_manual_stop_clicked,
        ).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        self._create_hold_button(
            frame,
            text="Right",
            row=1,
            column=2,
            command=(self._args.turn_speed, -self._args.turn_speed),
            color="#9ec5fe",
        )
        self._create_hold_button(
            frame,
            text="Reverse",
            row=2,
            column=1,
            command=(-self._args.speed, -self._args.speed),
            color="#f7c97f",
        )

    def _build_status_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Status", padding=8)
        frame.pack(fill="both", expand=True)
        self._status_text = tk.Text(
            frame,
            width=42,
            height=30,
            wrap="word",
            font=("Consolas", 10),
            bg="white",
        )
        self._status_text.pack(fill="both", expand=True)
        self._status_text.configure(state="disabled", cursor="arrow")

    def _create_hold_button(
        self,
        parent,
        *,
        text: str,
        row: int,
        column: int,
        command: tuple[float, float],
        color: str,
    ) -> None:
        button = tk.Button(
            parent,
            text=text,
            bg=color,
            activebackground="#dddddd",
        )
        button.grid(row=row, column=column, sticky="ew", padx=4, pady=4)
        button.bind(
            "<ButtonPress-1>",
            lambda _event, cmd=command, label=text.lower(): self._start_manual_command(cmd, label),
        )
        button.bind("<ButtonRelease-1>", self._on_global_button_release)

    def _start_manual_command(self, command: tuple[float, float], label: str) -> None:
        self._active_manual_command = command
        self._controller.clear_target(f"Manual {label} command active.")
        self._drive_state.set(*command)

    def _release_manual_command(self) -> None:
        if self._active_manual_command is None:
            return
        self._active_manual_command = None
        self._drive_state.stop()
        self._controller.set_message("Manual control released.")

    def _on_global_button_release(self, _event) -> None:
        self._release_manual_command()

    def _on_manual_stop_clicked(self) -> None:
        self._active_manual_command = None
        self._drive_state.stop()
        self._controller.clear_target("Manual stop requested.")

    def _on_go_clicked(self) -> None:
        try:
            target_x = float(self._target_x_var.get()) * self._input_to_vicon_scale
            target_y = float(self._target_y_var.get()) * self._input_to_vicon_scale
        except ValueError:
            self._controller.clear_target("Invalid target entry. Enter numeric X and Y.")
            return

        self._controller.set_target(target_x, target_y)

    def _on_stop_clicked(self) -> None:
        self._controller.clear_target("Stopped by user.")

    def _set_status_text(self, text: str) -> None:
        self._status_text.configure(state="normal")
        self._status_text.delete("1.0", "end")
        self._status_text.insert("1.0", text)
        self._status_text.configure(state="disabled")

    def _schedule_refresh(self) -> None:
        if self._closed:
            return
        self._refresh()
        if not self._closed:
            self._update_job = self._root.after(self._args.update_interval_ms, self._schedule_refresh)

    def _refresh(self) -> None:
        snapshot = self._shared_state.snapshot(None)
        visible = snapshot["visible"]
        obstacles = collect_obstacles(visible, self._selected_object_name)
        drive_command = self._drive_state.snapshot()
        teleop_status = self._teleop_client.snapshot()
        target_state = self._controller.snapshot()
        tuning_state = self._controller.tuning_snapshot()
        sample_rollout_paths = self._controller.sample_rollout_snapshot()

        self._ax.clear()
        self._ax.set_xlabel(f"X ({self._args.units})")
        self._ax.set_ylabel(f"Y ({self._args.units})")
        self._ax.set_title("Vicon MPPI Obstacle Controller (2D)")

        plotted_points = draw_room_2d(self._ax, self._room_center_xy, self._room_size_xy, self._args.units)
        names = sorted(visible)
        legend_handles: list[Line2D] = []

        for name in names:
            is_robot = name == self._selected_object_name
            color = "#1f77b4" if is_robot else "#d35400"
            legend_handles.append(
                Line2D([0], [0], color=color, marker="o", linestyle="-", linewidth=1.5, markersize=6, label=name)
            )
            pose = visible[name]["pose"]
            history_points = [(x * self._display_scale, y * self._display_scale) for x, y, _z in visible[name]["points"]]

            if history_points:
                self._ax.plot(
                    [point[0] for point in history_points],
                    [point[1] for point in history_points],
                    color=color,
                    linewidth=1.5,
                    alpha=0.75,
                )
                plotted_points.extend(history_points)

            plotted_points.extend(
                draw_pose_marker_2d(
                    self._ax,
                    pose=pose,
                    display_scale=self._display_scale,
                    color=color,
                    label=name,
                    body_size=self._robot_footprint,
                    axis_length=self._robot_axis_length,
                    heading_correction_deg=tuning_state.heading_offset_deg,
                )
            )

            if not is_robot:
                plotted_points.extend(
                    draw_obstacle_circle_2d(
                        self._ax,
                        x=pose.tx * self._display_scale,
                        y=pose.ty * self._display_scale,
                        radius=self._args.obstacle_radius * self._display_scale,
                        color="#c0392b",
                    )
                )

        if target_state.active or target_state.reached:
            target_x_display = target_state.target_x * self._display_scale
            target_y_display = target_state.target_y * self._display_scale
            self._ax.scatter(
                [target_x_display],
                [target_y_display],
                marker="*",
                s=160,
                color="#8e44ad",
                edgecolor="white",
                linewidths=0.8,
                zorder=6,
            )
            self._ax.text(target_x_display, target_y_display, " target", color="#6c3483")
            plotted_points.append((target_x_display, target_y_display))
            legend_handles.append(
                Line2D([0], [0], color="#8e44ad", marker="*", linestyle="None", markersize=10, label="target")
            )

            if self._selected_object_name in visible:
                pose = visible[self._selected_object_name]["pose"]
                self._ax.plot(
                    [pose.tx * self._display_scale, target_x_display],
                    [pose.ty * self._display_scale, target_y_display],
                    color="#8e44ad",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.5,
                )

        if sample_rollout_paths:
            rollout_alpha = min(0.26, max(0.04, 8.0 / max(len(sample_rollout_paths), 1)))
            for rollout_path in sample_rollout_paths:
                if not rollout_path:
                    continue
                path_points = [
                    (preview_x * self._display_scale, preview_y * self._display_scale)
                    for preview_x, preview_y, _preview_heading in rollout_path
                ]
                self._ax.plot(
                    [point[0] for point in path_points],
                    [point[1] for point in path_points],
                    color="#2ecc71",
                    linewidth=0.9,
                    alpha=rollout_alpha,
                    zorder=7,
                )
                plotted_points.extend(path_points)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#2ecc71",
                    linestyle="-",
                    linewidth=1.2,
                    alpha=0.75,
                    label="sample rollouts",
                )
            )

        if obstacles:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#c0392b",
                    linestyle=":",
                    linewidth=1.3,
                    label="obstacle radius",
                )
            )

        self._ax.scatter([0.0], [0.0], color="black", marker="x", s=40)
        plotted_points.append((0.0, 0.0))
        set_axes_equal_2d(self._ax, plotted_points, padding=self._args.axis_padding * self._display_scale)

        if legend_handles and len(legend_handles) <= 10:
            deduped_handles: dict[str, Line2D] = {}
            for handle in legend_handles:
                deduped_handles[handle.get_label()] = handle
            self._ax.legend(handles=list(deduped_handles.values()), loc="upper left", bbox_to_anchor=(1.02, 1.0))

        self._canvas.draw_idle()

        status_lines = build_status_lines(
            snapshot,
            self._selected_object_name,
            self._args.units,
            self._display_scale,
        )
        if snapshot["last_packet_time"] is None:
            status_lines.append("Waiting for UDP packets...")
        elif self._selected_object_name and self._selected_object_name not in visible:
            status_lines.append("Receiving packets, but the selected object is not visible.")

        target_x_status = target_state.target_x * self._display_scale
        target_y_status = target_state.target_y * self._display_scale
        robot_pose = visible.get(self._selected_object_name, {}).get("pose") if self._selected_object_name else None
        closest_obstacle_clearance: float | None = None
        if robot_pose is not None and obstacles:
            closest_obstacle_clearance = min(
                math.hypot(robot_pose.tx - obstacle.x, robot_pose.ty - obstacle.y) - self._args.obstacle_radius
                for obstacle in obstacles
            )

        status_lines.extend(
            [
                "",
                f"Drive backend: {teleop_status['backend']}",
                f"Drive endpoint: {teleop_status['endpoint']}",
                f"Control link: {'connected' if teleop_status['connected'] else 'disconnected'}",
                f"Drive command: ({drive_command.left:.2f}, {drive_command.right:.2f})",
                f"Controller mode: {target_state.mode}",
                f"Target ({self._args.units}): ({target_x_status:.3f}, {target_y_status:.3f})",
                f"Epsilon ({self._args.units}): {tuning_state.epsilon * self._display_scale:.3f}",
                f"Angle correction (deg): {tuning_state.heading_offset_deg:.1f}",
                f"MPPI: samples={self._args.num_samples}, horizon={self._args.horizon_steps}, temp={self._args.temperature:.2f}",
                f"Rollout preview paths: {len(sample_rollout_paths)}",
                f"Model: v={self._args.model_linear_speed:.1f} {self._args.units}/s, w={self._args.model_angular_speed:.2f} rad/s",
                f"Obstacles: {len(obstacles)} tracked, radius={self._args.obstacle_radius * self._display_scale:.1f} {self._args.units}",
                f"Obstacle band ({self._args.units}): {(self._args.obstacle_radius + self._args.obstacle_influence_radius) * self._display_scale:.1f}",
            ]
        )

        if target_state.last_distance is not None:
            status_lines.append(
                f"Distance to target ({self._args.units}): {target_state.last_distance * self._display_scale:.3f}"
            )
        if target_state.last_heading_error_deg is not None:
            status_lines.append(f"Heading error (deg): {target_state.last_heading_error_deg:.1f}")
        if closest_obstacle_clearance is not None:
            status_lines.append(
                f"Closest obstacle clearance ({self._args.units}): {closest_obstacle_clearance * self._display_scale:.3f}"
            )

        status_lines.extend(
            [
                f"Message: {target_state.message}",
                "Target entry uses the same units shown on the axes.",
            ]
        )
        if teleop_status["last_error"]:
            status_lines.append(f"Control error: {teleop_status['last_error']}")

        self._set_status_text("\n".join(status_lines))

    def run(self) -> bool:
        self._root.mainloop()
        return self._restart_requested

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        self._controller.stop()
        self._teleop_client.stop()
        self._receiver.stop()
        self._controller.join(timeout=1.0)
        self._teleop_client.join(timeout=1.0)
        self._receiver.join(timeout=1.0)

    def _close_window(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._update_job is not None:
            self._root.after_cancel(self._update_job)
            self._update_job = None
        self.shutdown()
        self._root.quit()
        self._root.destroy()

    def _on_back_to_setup(self) -> None:
        self._restart_requested = True
        self._close_window()

    def _on_close(self) -> None:
        self._restart_requested = False
        self._close_window()


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    startup_initial_values = vars(args).copy()
    saved_defaults = _load_saved_defaults(DEFAULTS_CONFIG_PATH)
    explicit_cli_dests = _explicit_cli_destinations(parser, sys.argv[1:])
    for key, value in saved_defaults.items():
        if key in startup_initial_values and key not in explicit_cli_dests:
            startup_initial_values[key] = value

    while True:
        startup_values = show_mppi_startup_config_tk(
            title="MPPI Obstacle 2D Setup",
            initial_values=startup_initial_values,
            include_obstacles=True,
            config_path=str(DEFAULTS_CONFIG_PATH),
        )
        if startup_values is None:
            return
        for key, value in startup_values.items():
            setattr(args, key, value)
        startup_initial_values = vars(args).copy()

        selected_object_name = args.object_name or None
        display_scale = 1.0 if args.units == "mm" else 0.001
        room_center_xy = (
            args.room_center[0] * display_scale,
            args.room_center[1] * display_scale,
        )
        room_size_xy = (
            args.room_size[0] * display_scale,
            args.room_size[1] * display_scale,
        )
        robot_footprint = args.robot_footprint * display_scale
        robot_axis_length = args.robot_axis_length * display_scale
        input_to_vicon_scale = 1.0 if args.units == "mm" else 1000.0

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
        try:
            teleop_client = create_drive_client(
                drive_backend=args.drive_backend,
                drive_state=drive_state,
                send_rate_hz=args.send_rate_hz,
                verbose=args.verbose,
                jetbot_host=args.jetbot_host,
                jetbot_port=args.jetbot_port,
                serial_port=args.serial_port,
                serial_baud=args.serial_baud,
                serial_max_pwm=args.serial_max_pwm,
            )
        except ValueError as exc:
            parser.error(str(exc))

        controller = ObstacleAwareMPPIController(
            shared_state=shared_state,
            selected_name=selected_object_name or args.object_name,
            drive_state=drive_state,
            speed=args.speed,
            turn_speed=args.turn_speed,
            min_forward_speed=args.min_forward_speed,
            slow_near_target=args.slow_near_target,
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
            obstacle_radius=args.obstacle_radius,
            obstacle_influence_radius=args.obstacle_influence_radius,
            obstacle_cost_weight=args.obstacle_cost_weight,
            obstacle_collision_cost=args.obstacle_collision_cost,
        )

        receiver.start()
        teleop_client.start()
        controller.start()

        app = ObstacleController2DApp(
            args=args,
            shared_state=shared_state,
            receiver=receiver,
            drive_state=drive_state,
            teleop_client=teleop_client,
            controller=controller,
            selected_object_name=selected_object_name,
            display_scale=display_scale,
            room_center_xy=room_center_xy,
            room_size_xy=room_size_xy,
            robot_footprint=robot_footprint,
            robot_axis_length=robot_axis_length,
            input_to_vicon_scale=input_to_vicon_scale,
        )

        try:
            restart_requested = app.run()
        finally:
            app.shutdown()

        if not restart_requested:
            return


if __name__ == "__main__":
    main()
