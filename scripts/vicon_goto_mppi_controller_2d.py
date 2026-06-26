"""
2D driving-plane version of the MPPI go-to controller.

This runs the same MPPI control loop as scripts/vicon_goto_mppi_controller.py,
but renders only the XY driving plane instead of a full 3D room view.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, CheckButtons, TextBox

from drive_backends import DriveState, create_drive_client
from mppi_startup_ui import show_mppi_startup_config
from vicon_goto_mppi_controller import MPPIController, make_parser as make_base_parser
from vicon_udp_viewer import (
    SharedState,
    ViconReceiver,
    build_status_lines,
    rotate_vector,
    rotation_matrix_xyz,
)


def make_parser():
    parser = make_base_parser()
    parser.description = "Receive the Vicon UDP stream, show the XY driving plane, and drive the selected robot to a target with an MPPI controller."
    return parser


def set_axes_equal_2d(ax, points: list[tuple[float, float]], padding: float) -> None:
    if not points:
        radius = max(padding, 1.0)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    half_span = max(
        (x_max - x_min) / 2.0,
        (y_max - y_min) / 2.0,
        padding,
    )

    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(y_center - half_span, y_center + half_span)
    ax.set_aspect("equal", adjustable="box")


def room_corners_2d(
    room_center: tuple[float, float],
    room_size: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    cx, cy = room_center
    sx, sy = room_size
    hx, hy = sx / 2.0, sy / 2.0
    return {
        "lb": (cx - hx, cy - hy),
        "rb": (cx + hx, cy - hy),
        "rt": (cx + hx, cy + hy),
        "lt": (cx - hx, cy + hy),
    }


def draw_room_2d(
    ax,
    room_center: tuple[float, float],
    room_size: tuple[float, float],
    units: str,
) -> list[tuple[float, float]]:
    corners = room_corners_2d(room_center, room_size)
    loop = [corners["lb"], corners["rb"], corners["rt"], corners["lt"], corners["lb"]]
    ax.plot(
        [point[0] for point in loop],
        [point[1] for point in loop],
        color="#7f8c8d",
        linewidth=1.2,
        alpha=0.75,
    )
    ax.fill(
        [point[0] for point in loop],
        [point[1] for point in loop],
        color="#7fb3d5",
        alpha=0.05,
    )

    x_ticks = 4
    y_ticks = 4
    x_min, x_max = corners["lb"][0], corners["rb"][0]
    y_min, y_max = corners["lb"][1], corners["lt"][1]

    for index in range(1, x_ticks):
        x = x_min + (x_max - x_min) * index / x_ticks
        ax.plot([x, x], [y_min, y_max], color="#d5d8dc", linewidth=0.8, alpha=0.5)
    for index in range(1, y_ticks):
        y = y_min + (y_max - y_min) * index / y_ticks
        ax.plot([x_min, x_max], [y, y], color="#d5d8dc", linewidth=0.8, alpha=0.5)

    ax.text(
        corners["lb"][0],
        corners["lt"][1],
        f"Room ({units})",
        color="#566573",
        fontsize=9,
        va="bottom",
    )
    return list(corners.values())


def draw_pose_marker_2d(
    ax,
    *,
    pose,
    display_scale: float,
    color,
    label: str,
    body_size: float,
    axis_length: float,
    heading_correction_deg: float = 0.0,
) -> list[tuple[float, float]]:
    center = (pose.tx * display_scale, pose.ty * display_scale)
    rotation = rotation_matrix_xyz(pose.rx, pose.ry, pose.rz)

    half_body = body_size / 2.0
    local_corners = [
        (-half_body, -half_body, 0.0),
        (half_body, -half_body, 0.0),
        (half_body, half_body, 0.0),
        (-half_body, half_body, 0.0),
    ]
    world_corners = []
    for corner in local_corners:
        rotated = rotate_vector(rotation, corner)
        world_corners.append((center[0] + rotated[0], center[1] + rotated[1]))

    loop = world_corners + [world_corners[0]]
    ax.plot(
        [point[0] for point in loop],
        [point[1] for point in loop],
        color=color,
        linewidth=1.8,
    )

    physical_forward = rotate_vector(rotation, (0.0, axis_length, 0.0))
    heading_x = physical_forward[0]
    heading_y = physical_forward[1]
    heading_norm = math.hypot(heading_x, heading_y)
    if heading_norm <= 1e-9:
        heading_x, heading_y = 1.0, 0.0
        heading_norm = 1.0
    heading_x /= heading_norm
    heading_y /= heading_norm

    if abs(heading_correction_deg) > 1e-9:
        correction_rad = math.radians(heading_correction_deg)
        cos_theta = math.cos(correction_rad)
        sin_theta = math.sin(correction_rad)
        heading_x, heading_y = (
            heading_x * cos_theta - heading_y * sin_theta,
            heading_x * sin_theta + heading_y * cos_theta,
        )

    heading_tip = (
        center[0] + heading_x * axis_length,
        center[1] + heading_y * axis_length,
    )
    ax.annotate(
        "",
        xy=heading_tip,
        xytext=center,
        arrowprops={"arrowstyle": "->", "color": "#111111", "linewidth": 2.4},
    )
    ax.scatter([center[0]], [center[1]], color=[color], s=50, zorder=5)
    ax.text(center[0], center[1], f" {label}", color=color)

    return world_corners + [heading_tip, center]


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    startup_values = show_mppi_startup_config(
        title="MPPI Go-To 2D Setup",
        initial_values=vars(args).copy(),
        include_obstacles=False,
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

    controller = MPPIController(
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

    color_cycle = list(plt.get_cmap("tab10").colors)
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
        {"label": "Left", "rect": [0.18, 0.03, 0.11, 0.055], "command": (-args.turn_speed, args.turn_speed), "color": "#9ec5fe"},
        {"label": "Forward", "rect": [0.31, 0.03, 0.11, 0.055], "command": (args.speed, args.speed), "color": "#8fd19e"},
        {"label": "Stop", "rect": [0.44, 0.03, 0.11, 0.055], "command": None, "color": "#f5a3a3"},
        {"label": "Right", "rect": [0.57, 0.03, 0.11, 0.055], "command": (args.turn_speed, -args.turn_speed), "color": "#9ec5fe"},
        {"label": "Reverse", "rect": [0.70, 0.03, 0.11, 0.055], "command": (-args.speed, -args.speed), "color": "#f7c97f"},
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
        snapshot = shared_state.snapshot(selected_object_name)
        visible = snapshot["visible"]
        drive_command = drive_state.snapshot()
        teleop_status = teleop_client.snapshot()
        target_state = controller.snapshot()
        tuning_state = controller.tuning_snapshot()
        sample_heading_predictions = controller.sample_heading_snapshot()

        ax.cla()
        ax.set_xlabel(f"X ({args.units})")
        ax.set_ylabel(f"Y ({args.units})")
        ax.set_title("Vicon MPPI Go-To Controller (2D)")

        plotted_points = draw_room_2d(ax, room_center_xy, room_size_xy, args.units)
        names = sorted(visible)
        legend_handles: list[Line2D] = []

        for index, name in enumerate(names):
            color = color_cycle[index % len(color_cycle)]
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
                Line2D([0], [0], color="#16a085", linestyle="-", linewidth=1.2, alpha=0.7, label="sample headings")
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
