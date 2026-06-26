"""
2D driving-plane version of the obstacle-aware MPPI go-to controller.

This runs the same obstacle-aware MPPI control loop as
scripts/vicon_goto_mppi_obstacle_controller.py, but renders only the XY driving
plane instead of a full 3D room view.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, CheckButtons, TextBox

from drive_backends import DriveState, create_drive_client
from mppi_startup_ui import show_mppi_startup_config
from vicon_goto_mppi_controller_2d import draw_pose_marker_2d, draw_room_2d, set_axes_equal_2d
from vicon_goto_mppi_obstacle_controller import (
    ObstacleAwareMPPIController,
    collect_obstacles,
    make_parser as make_base_parser,
)
from vicon_udp_viewer import SharedState, ViconReceiver, build_status_lines


def make_parser():
    parser = make_base_parser()
    parser.description = (
        "Receive the Vicon UDP stream, show the XY driving plane, and drive the "
        "selected robot to a target with an obstacle-aware MPPI controller."
    )
    return parser


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


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    startup_values = show_mppi_startup_config(
        title="MPPI Obstacle 2D Setup",
        initial_values=vars(args).copy(),
        include_obstacles=True,
    )
    if startup_values is None:
        return
    for key, value in startup_values.items():
        setattr(args, key, value)

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

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_axes([0.26, 0.22, 0.68, 0.70])
    status_ax = fig.add_axes([0.03, 0.22, 0.20, 0.70])
    status_ax.axis("off")

    fig.text(0.08, 0.185, "Go-To Target", fontsize=11, fontweight="bold")
    fig.text(0.08, 0.162, f"Target X ({args.units})", fontsize=9)
    fig.text(0.26, 0.162, f"Target Y ({args.units})", fontsize=9)
    fig.text(0.50, 0.095, "Manual Drive (press and hold)", fontsize=11, fontweight="bold", ha="center")

    x_box_ax = fig.add_axes([0.08, 0.105, 0.14, 0.055])
    y_box_ax = fig.add_axes([0.26, 0.105, 0.14, 0.055])
    go_button_ax = fig.add_axes([0.45, 0.105, 0.11, 0.055])
    stop_button_ax = fig.add_axes([0.60, 0.105, 0.16, 0.055])
    sample_heading_toggle_ax = fig.add_axes([0.78, 0.095, 0.16, 0.08])

    x_text = TextBox(x_box_ax, "", initial="0")
    y_text = TextBox(y_box_ax, "", initial="0")
    go_button = Button(go_button_ax, "Go", color="#8fd19e", hovercolor="#d5f5e3")
    stop_button = Button(stop_button_ax, "Stop Go-To", color="#f5a3a3", hovercolor="#fadbd8")
    sample_heading_toggle = CheckButtons(sample_heading_toggle_ax, ["Show Sample\nHeadings"], [False])
    for label in sample_heading_toggle.labels:
        label.set_fontsize(9)
    go_button.label.set_fontsize(10)
    stop_button.label.set_fontsize(10)

    active_manual_command: tuple[float, float] | None = None
    show_sample_headings = False

    def on_go_clicked(_event) -> None:
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

    for spec in manual_button_specs:
        button_ax = fig.add_axes(spec["rect"])
        button = Button(button_ax, spec["label"], color=spec["color"], hovercolor="#dddddd")
        button.label.set_fontsize(10)
        manual_axes_to_spec[button_ax] = spec

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

    def on_sample_heading_toggle(_label: str) -> None:
        nonlocal show_sample_headings
        show_sample_headings = sample_heading_toggle.get_status()[0]

    go_button.on_clicked(on_go_clicked)
    stop_button.on_clicked(on_stop_clicked)
    sample_heading_toggle.on_clicked(on_sample_heading_toggle)
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)

    def update_plot(_frame_index: int) -> None:
        snapshot = shared_state.snapshot(None)
        visible = snapshot["visible"]
        obstacles = collect_obstacles(visible, selected_object_name)
        drive_command = drive_state.snapshot()
        teleop_status = teleop_client.snapshot()
        target_state = controller.snapshot()
        tuning_state = controller.tuning_snapshot()
        sample_heading_predictions = controller.sample_heading_snapshot()

        ax.cla()
        ax.set_xlabel(f"X ({args.units})")
        ax.set_ylabel(f"Y ({args.units})")
        ax.set_title("Vicon MPPI Obstacle Controller (2D)")

        plotted_points = draw_room_2d(ax, room_center_xy, room_size_xy, args.units)
        names = sorted(visible)
        legend_handles: list[Line2D] = []

        for name in names:
            is_robot = name == selected_object_name
            color = "#1f77b4" if is_robot else "#d35400"
            legend_handles.append(
                Line2D([0], [0], color=color, marker="o", linestyle="-", linewidth=1.5, markersize=6, label=name)
            )
            pose = visible[name]["pose"]
            history_points = [
                (x * display_scale, y * display_scale)
                for x, y, _z in visible[name]["points"]
            ]

            if history_points:
                ax.plot(
                    [point[0] for point in history_points],
                    [point[1] for point in history_points],
                    color=color,
                    linewidth=1.5,
                    alpha=0.75,
                )
                plotted_points.extend(history_points)

            plotted_points.extend(
                draw_pose_marker_2d(
                    ax,
                    pose=pose,
                    display_scale=display_scale,
                    color=color,
                    label=name,
                    body_size=robot_footprint,
                    axis_length=robot_axis_length,
                    heading_correction_deg=tuning_state.heading_offset_deg,
                )
            )

            if not is_robot:
                plotted_points.extend(
                    draw_obstacle_circle_2d(
                        ax,
                        x=pose.tx * display_scale,
                        y=pose.ty * display_scale,
                        radius=args.obstacle_radius * display_scale,
                        color="#c0392b",
                    )
                )

        if target_state.active or target_state.reached:
            target_x_display = target_state.target_x * display_scale
            target_y_display = target_state.target_y * display_scale
            ax.scatter(
                [target_x_display],
                [target_y_display],
                marker="*",
                s=160,
                color="#8e44ad",
                edgecolor="white",
                linewidths=0.8,
                zorder=6,
            )
            ax.text(target_x_display, target_y_display, " target", color="#6c3483")
            plotted_points.append((target_x_display, target_y_display))
            legend_handles.append(
                Line2D([0], [0], color="#8e44ad", marker="*", linestyle="None", markersize=10, label="target")
            )

            if selected_object_name in visible:
                pose = visible[selected_object_name]["pose"]
                ax.plot(
                    [pose.tx * display_scale, target_x_display],
                    [pose.ty * display_scale, target_y_display],
                    color="#8e44ad",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.5,
                )

        if show_sample_headings and sample_heading_predictions:
            sample_heading_length = max(robot_axis_length * 0.55, 1.0)
            sample_alpha = min(0.25, max(0.04, 12.0 / max(len(sample_heading_predictions), 1)))
            for preview_x, preview_y, preview_heading in sample_heading_predictions:
                start_x = preview_x * display_scale
                start_y = preview_y * display_scale
                end_x = start_x + sample_heading_length * math.cos(preview_heading)
                end_y = start_y + sample_heading_length * math.sin(preview_heading)
                ax.plot(
                    [start_x, end_x],
                    [start_y, end_y],
                    color="#16a085",
                    linewidth=1.0,
                    alpha=sample_alpha,
                )
                plotted_points.extend([(start_x, start_y), (end_x, end_y)])
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#16a085",
                    linestyle="-",
                    linewidth=1.2,
                    alpha=0.7,
                    label="sample headings",
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

        ax.scatter([0.0], [0.0], color="black", marker="x", s=40)
        plotted_points.append((0.0, 0.0))
        set_axes_equal_2d(ax, plotted_points, padding=args.axis_padding * display_scale)

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
        robot_pose = visible.get(selected_object_name, {}).get("pose") if selected_object_name else None
        closest_obstacle_clearance: float | None = None
        if robot_pose is not None and obstacles:
            closest_obstacle_clearance = min(
                math.hypot(robot_pose.tx - obstacle.x, robot_pose.ty - obstacle.y) - args.obstacle_radius
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
                f"Target ({args.units}): ({target_x_status:.3f}, {target_y_status:.3f})",
                f"Epsilon ({args.units}): {tuning_state.epsilon * display_scale:.3f}",
                f"Angle correction (deg): {tuning_state.heading_offset_deg:.1f}",
                f"MPPI: samples={args.num_samples}, horizon={args.horizon_steps}, temp={args.temperature:.2f}",
                f"Model: v={args.model_linear_speed:.1f} {args.units}/s, w={args.model_angular_speed:.2f} rad/s",
                f"Obstacles: {len(obstacles)} tracked, radius={args.obstacle_radius * display_scale:.1f} {args.units}",
                f"Obstacle band ({args.units}): {(args.obstacle_radius + args.obstacle_influence_radius) * display_scale:.1f}",
            ]
        )

        if target_state.last_distance is not None:
            status_lines.append(f"Distance to target ({args.units}): {target_state.last_distance * display_scale:.3f}")
        if target_state.last_heading_error_deg is not None:
            status_lines.append(f"Heading error (deg): {target_state.last_heading_error_deg:.1f}")
        if closest_obstacle_clearance is not None:
            status_lines.append(
                f"Closest obstacle clearance ({args.units}): {closest_obstacle_clearance * display_scale:.3f}"
            )

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
