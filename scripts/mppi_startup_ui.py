from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox


@dataclass(frozen=True)
class StartupField:
    key: str
    label: str
    kind: str = "str"
    choices: tuple[str, ...] = ()


def _parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise ValueError("use true/false")


def _parse_float3(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("use three comma-separated numbers")
    return tuple(float(part) for part in parts)


def _parse_value(field: StartupField, raw_text: str):
    if field.kind == "str":
        return raw_text.strip()
    if field.kind == "int":
        return int(raw_text.strip())
    if field.kind == "float":
        return float(raw_text.strip())
    if field.kind == "bool":
        return _parse_bool(raw_text)
    if field.kind == "choice":
        value = raw_text.strip()
        if value not in field.choices:
            raise ValueError(f"choose one of: {', '.join(field.choices)}")
        return value
    if field.kind == "float3":
        return _parse_float3(raw_text)
    raise ValueError(f"unsupported field kind: {field.kind}")


def _format_value(field: StartupField, value: object) -> str:
    if field.kind == "bool":
        return "true" if bool(value) else "false"
    if field.kind == "float3":
        values = tuple(value)
        return ", ".join(f"{float(item):g}" for item in values)
    return str(value)


def _common_connection_fields() -> list[StartupField]:
    return [
        StartupField("object_name", "Object Name"),
        StartupField("source_ip", "Vicon Source IP"),
        StartupField("bind_host", "Bind Host"),
        StartupField("port", "UDP Port", "int"),
        StartupField("drive_backend", "Drive Backend", "choice", ("jetbot-socket", "arduino-bluetooth")),
        StartupField("jetbot_host", "JetBot Host"),
        StartupField("jetbot_port", "JetBot Port", "int"),
        StartupField("serial_port", "Serial Port"),
        StartupField("serial_baud", "Serial Baud", "int"),
        StartupField("serial_max_pwm", "Serial Max PWM", "int"),
        StartupField("send_rate_hz", "Send Rate (Hz)", "float"),
    ]


def _common_display_fields() -> list[StartupField]:
    return [
        StartupField("units", "Display Units", "choice", ("mm", "m")),
        StartupField("history", "History", "int"),
        StartupField("stale_after", "Stale After (s)", "float"),
        StartupField("update_interval_ms", "Update Interval (ms)", "int"),
        StartupField("buffer_size", "Buffer Size", "int"),
        StartupField("axis_padding", "Axis Padding", "float"),
        StartupField("elevation_deg", "Elevation (deg)", "float"),
        StartupField("azimuth_deg", "Azimuth (deg)", "float"),
        StartupField("room_size", "Room Size X,Y,Z", "float3"),
        StartupField("room_center", "Room Center X,Y,Z", "float3"),
        StartupField("robot_footprint", "Robot Footprint", "float"),
        StartupField("robot_axis_length", "Robot Axis Length", "float"),
    ]


def _common_control_fields() -> list[StartupField]:
    return [
        StartupField("speed", "Speed", "float"),
        StartupField("turn_speed", "Turn Speed", "float"),
        StartupField("min_forward_speed", "Min Forward Speed", "float"),
        StartupField("epsilon", "Epsilon", "float"),
        StartupField("slow_radius", "Slow Radius", "float"),
        StartupField("slow_near_target", "Slow Near Target", "bool"),
        StartupField("heading_gain", "Heading Gain", "float"),
        StartupField("forward_offset_deg", "Angle Correction (deg)", "float"),
        StartupField("control_rate_hz", "Control Rate (Hz)", "float"),
    ]


def _common_mppi_fields_a() -> list[StartupField]:
    return [
        StartupField("horizon_steps", "Horizon Steps", "int"),
        StartupField("num_samples", "Num Samples", "int"),
        StartupField("temperature", "Temperature", "float"),
        StartupField("noise_std", "Noise Std", "float"),
        StartupField("model_linear_speed", "Model Linear Speed", "float"),
        StartupField("model_angular_speed", "Model Angular Speed", "float"),
        StartupField("heuristic_blend", "Heuristic Blend", "float"),
    ]


def _common_mppi_fields_b() -> list[StartupField]:
    return [
        StartupField("distance_cost_weight", "Distance Weight", "float"),
        StartupField("heading_cost_weight", "Heading Weight", "float"),
        StartupField("terminal_distance_cost_weight", "Term Dist Weight", "float"),
        StartupField("terminal_heading_cost_weight", "Term Heading Weight", "float"),
        StartupField("control_effort_weight", "Effort Weight", "float"),
        StartupField("control_slew_weight", "Slew Weight", "float"),
        StartupField("reverse_cost_weight", "Reverse Weight", "float"),
    ]


def _obstacle_fields() -> list[StartupField]:
    return [
        StartupField("obstacle_radius", "Obstacle Radius", "float"),
        StartupField("obstacle_influence_radius", "Obstacle Influence", "float"),
        StartupField("obstacle_cost_weight", "Obstacle Weight", "float"),
        StartupField("obstacle_collision_cost", "Collision Cost", "float"),
    ]


def _debug_fields() -> list[StartupField]:
    return [StartupField("verbose", "Verbose", "bool")]


def _build_columns(include_obstacles: bool) -> list[list[tuple[str, list[StartupField]]]]:
    columns: list[list[tuple[str, list[StartupField]]]] = [
        [("Connection", _common_connection_fields())],
        [("Display", _common_display_fields())],
        [("Control", _common_control_fields())],
        [("MPPI A", _common_mppi_fields_a())],
        [("MPPI B", _common_mppi_fields_b())],
    ]
    if include_obstacles:
        columns.append(
            [
                ("Obstacles", _obstacle_fields()),
                ("Debug", _debug_fields()),
            ]
        )
    else:
        columns[-1].append(("Debug", _debug_fields()))
    return columns


def _field_lookup(columns: list[list[tuple[str, list[StartupField]]]]) -> dict[str, StartupField]:
    return {
        field.key: field
        for section_group in columns
        for _, fields in section_group
        for field in fields
    }


def show_mppi_startup_config(
    *,
    title: str,
    initial_values: dict[str, object],
    include_obstacles: bool,
) -> dict[str, object] | None:
    columns = _build_columns(include_obstacles)

    fig = plt.figure(figsize=(19, 10))
    fig.text(0.03, 0.965, title, fontsize=16, fontweight="bold")
    fig.text(
        0.03,
        0.935,
        "Set startup parameters here, then click Next. Use true/false for booleans and X, Y, Z for room vectors.",
        fontsize=10,
    )
    fig.text(
        0.03,
        0.913,
        "Choices: drive backend = jetbot-socket or arduino-bluetooth. Display units = mm or m.",
        fontsize=9,
        color="#555555",
    )

    textboxes: dict[str, TextBox] = {}
    error_text = fig.text(0.03, 0.04, "", fontsize=10, color="#b00020")
    result: dict[str, object] = {"accepted": False, "values": None}

    left_margin = 0.03
    right_margin = 0.03
    column_gap = 0.02
    column_width = (1.0 - left_margin - right_margin - column_gap * (len(columns) - 1)) / len(columns)
    field_height = 0.032
    label_gap = 0.004
    row_pitch = 0.058
    section_header_gap = 0.062
    section_footer_gap = 0.022

    for column_index, section_group in enumerate(columns):
        left = left_margin + column_index * (column_width + column_gap)
        cursor_y = 0.875
        for section_name, fields in section_group:
            fig.text(left, cursor_y, section_name, fontsize=11, fontweight="bold")
            cursor_y -= section_header_gap
            for field in fields:
                fig.text(left, cursor_y + field_height + label_gap, field.label, fontsize=8)
                box_ax = fig.add_axes([left, cursor_y, column_width - 0.01, field_height])
                textboxes[field.key] = TextBox(
                    box_ax,
                    "",
                    initial=_format_value(field, initial_values[field.key]),
                )
                cursor_y -= row_pitch
            cursor_y -= section_footer_gap

    next_button_ax = fig.add_axes([0.74, 0.03, 0.10, 0.05])
    cancel_button_ax = fig.add_axes([0.86, 0.03, 0.10, 0.05])
    next_button = Button(next_button_ax, "Next", color="#8fd19e", hovercolor="#d5f5e3")
    cancel_button = Button(cancel_button_ax, "Cancel", color="#f5a3a3", hovercolor="#fadbd8")
    next_button.label.set_fontsize(11)
    cancel_button.label.set_fontsize(11)

    field_lookup = _field_lookup(columns)

    def on_next_clicked(_event) -> None:
        values: dict[str, object] = {}
        try:
            for key, textbox in textboxes.items():
                values[key] = _parse_value(field_lookup[key], textbox.text)
            if values["drive_backend"] == "arduino-bluetooth" and not str(values["serial_port"]).strip():
                raise ValueError("Serial Port is required for the arduino-bluetooth backend.")
        except ValueError as exc:
            error_text.set_text(f"Invalid startup configuration: {exc}")
            fig.canvas.draw_idle()
            return

        result["accepted"] = True
        result["values"] = values
        plt.close(fig)

    def on_cancel_clicked(_event) -> None:
        plt.close(fig)

    next_button.on_clicked(on_next_clicked)
    cancel_button.on_clicked(on_cancel_clicked)
    plt.show()

    if not result["accepted"]:
        return None
    return result["values"]


def show_mppi_startup_config_tk(
    *,
    title: str,
    initial_values: dict[str, object],
    include_obstacles: bool,
    config_path: str | None = None,
) -> dict[str, object] | None:
    columns = _build_columns(include_obstacles)
    field_lookup = _field_lookup(columns)

    result: dict[str, object] = {"accepted": False, "values": None}
    root = tk.Tk()
    root.title(title)
    root.geometry("1900x980")
    root.minsize(1500, 860)
    root.configure(padx=14, pady=12)
    try:
        root.state("zoomed")
    except tk.TclError:
        pass

    header_frame = ttk.Frame(root)
    header_frame.pack(fill="x", pady=(0, 10))
    ttk.Label(header_frame, text=title, font=("Segoe UI", 18, "bold")).pack(anchor="w")
    ttk.Label(
        header_frame,
        text="Set startup parameters here, then click Next. Use true/false for booleans and X, Y, Z for room vectors.",
        font=("Segoe UI", 11),
    ).pack(anchor="w", pady=(4, 0))
    ttk.Label(
        header_frame,
        text="Choices: drive backend = jetbot-socket or arduino-bluetooth. Display units = mm or m.",
        foreground="#555555",
    ).pack(anchor="w", pady=(2, 0))

    content_frame = ttk.Frame(root)
    content_frame.pack(fill="both", expand=True)
    textvars: dict[str, tk.StringVar] = {}
    first_entry: ttk.Entry | None = None

    for column_index, section_group in enumerate(columns):
        column_frame = ttk.Frame(content_frame)
        column_frame.grid(row=0, column=column_index, sticky="nsew", padx=(0, 12) if column_index < len(columns) - 1 else 0)
        content_frame.grid_columnconfigure(column_index, weight=1)

        for section_name, fields in section_group:
            section_frame = ttk.LabelFrame(column_frame, text=section_name, padding=(10, 8))
            section_frame.pack(fill="x", pady=(0, 10))
            section_frame.grid_columnconfigure(0, weight=1)

            for field_index, field in enumerate(fields):
                row_base = field_index * 2
                ttk.Label(section_frame, text=field.label).grid(row=row_base, column=0, sticky="w")
                variable = tk.StringVar(value=_format_value(field, initial_values[field.key]))
                textvars[field.key] = variable
                entry = ttk.Entry(section_frame, textvariable=variable, width=28)
                entry.grid(row=row_base + 1, column=0, sticky="ew", pady=(0, 8))
                if first_entry is None:
                    first_entry = entry

    footer_frame = ttk.Frame(root)
    footer_frame.pack(fill="x", pady=(4, 0))
    status_var = tk.StringVar(value="")
    status_label = ttk.Label(footer_frame, textvariable=status_var, foreground="#b00020")
    status_label.pack(side="left", fill="x", expand=True)
    ttk.Button(footer_frame, text="Next", command=lambda: on_next_clicked()).pack(side="right", padx=(8, 0))
    ttk.Button(footer_frame, text="Cancel", command=lambda: on_cancel_clicked()).pack(side="right")
    if config_path:
        ttk.Button(footer_frame, text="Save as Default", command=lambda: on_save_defaults_clicked()).pack(
            side="right",
            padx=(8, 0),
        )

    def parse_current_values() -> dict[str, object]:
        values: dict[str, object] = {}
        for key, variable in textvars.items():
            values[key] = _parse_value(field_lookup[key], variable.get())
        if values["drive_backend"] == "arduino-bluetooth" and not str(values["serial_port"]).strip():
            raise ValueError("Serial Port is required for the arduino-bluetooth backend.")
        return values

    def on_next_clicked() -> None:
        try:
            values = parse_current_values()
        except ValueError as exc:
            status_var.set(f"Invalid startup configuration: {exc}")
            status_label.configure(foreground="#b00020")
            return

        result["accepted"] = True
        result["values"] = values
        root.destroy()

    def on_save_defaults_clicked() -> None:
        try:
            values = parse_current_values()
        except ValueError as exc:
            status_var.set(f"Invalid startup configuration: {exc}")
            status_label.configure(foreground="#b00020")
            return

        try:
            path = Path(config_path) if config_path else None
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(values, indent=2), encoding="utf-8")
        except OSError as exc:
            status_var.set(f"Could not save defaults: {exc}")
            status_label.configure(foreground="#b00020")
            return

        status_var.set(f"Saved defaults to {path}")
        status_label.configure(foreground="#1f6f43")

    def on_cancel_clicked() -> None:
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_cancel_clicked)
    root.bind("<Return>", lambda _event: on_next_clicked())
    root.bind("<Escape>", lambda _event: on_cancel_clicked())
    if first_entry is not None:
        root.after(10, first_entry.focus_set)
    root.mainloop()

    if not result["accepted"]:
        return None
    return result["values"]
