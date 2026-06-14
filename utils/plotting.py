from pathlib import Path

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from models.regression import predict
from utils.metrics import r2_score, rmse


def plot_model_comparison(
    x,
    y,
    results,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    equation_text: str | None = None,
    r2_value: float | None = None,
    rmse_value: float | None = None,
    mape_value: float | None = None,
    data_label: str = "Datos experimentales",
    point_labels: np.ndarray | None = None,
    fines: np.ndarray | None = None,
    embed_mode: bool = False,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_line = np.linspace(np.min(x), np.max(x), 200)
    fig = go.Figure()
    
    # Verificar si hay labels para colorear por prefijo (distrito)
    has_labels = point_labels is not None and len(point_labels) > 0 and len(point_labels) == len(x)
    prefix_counts = None
    prefix_order = None
    prefix_color_map = None
    if has_labels:
        # Extraer prefijos (ej: "CH" de "CH-1", "VEN" de "VEN-2")
        prefix_dict = {}
        for i, label in enumerate(point_labels):
            label_str = str(label).strip()
            match = re.match(r"([A-Z]+)", label_str)
            prefix = match.group(1) if match else "OTRO"

            if prefix not in prefix_dict:
                prefix_dict[prefix] = {"indices": [], "labels": []}
            prefix_dict[prefix]["indices"].append(i)
            prefix_dict[prefix]["labels"].append(label_str)
        prefix_counts = {prefix: len(data["indices"]) for prefix, data in prefix_dict.items()}
        prefix_order = list(prefix_dict.keys())

        # Paleta de colores por prefijo (distritos)
        color_palette = [
            "#1f4e79",  # Azul oscuro
            "#c00000",  # Rojo
            "#70ad47",  # Verde
            "#ffc000",  # Naranja
            "#5b9bd5",  # Azul claro
            "#ed7d31",  # Naranja oscuro
            "#a5a5a5",  # Gris
            "#44546a",  # Gris azulado
            "#e2efda",  # Verde claro
            "#fce4d6",  # Salmón
        ]

        # Agregar un trace por prefijo con color diferente
        prefix_color_map = {}
        for idx, (prefix, data) in enumerate(prefix_dict.items()):
            indices = np.array(data["indices"])
            labels = data["labels"]
            color = color_palette[idx % len(color_palette)]
            prefix_color_map[prefix] = color

            fig.add_trace(
                go.Scatter(
                    x=x[indices],
                    y=y[indices],
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    textfont={"size": 10, "color": color},
                    name=prefix,  # Aparecerá en la leyenda
                    marker={
                        "color": color,
                        "size": 9,
                        "line": {"color": "#ffffff", "width": 1.2},
                        "opacity": 0.95,
                    },
                    hovertemplate="%{text}<br>N60: %{x:.2f}<br>φ: %{y:.2f}<extra></extra>",
                )
            )
        
        # Agregar barras de dispersión por grupo con el color del grupo
        best_pred = None
        if len(results) > 0:
            try:
                best_pred = predict(results[0], x)
            except ValueError:
                if results[0].model_type == "sqrt_fines" and fines is not None:
                    params = results[0].params
                    best_pred = params["a"] * np.sqrt(x) + params["b"] * fines + params["c"]
                else:
                    best_pred = None
            if best_pred is not None:
                for prefix, data in prefix_dict.items():
                    indices = np.array(data["indices"])
                    color = prefix_color_map[prefix]
                    line_x = []
                    line_y = []
                    for i in indices:
                        line_x.extend([x[i], x[i], None])
                        line_y.extend([y[i], best_pred[i], None])
                    fig.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_y,
                            mode="lines",
                            name=f"Dispersión {prefix}",
                            line={"color": color, "width": 1.8, "dash": "dot"},
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

    else:
        # Si no hay labels, agregar todos los puntos con un color
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=data_label,
                marker={
                    "color": "#1f4e79",
                    "size": 11,
                    "symbol": "circle",
                    "line": {"color": "#ffffff", "width": 1.8},
                    "opacity": 0.95,
                },
                hovertemplate="N60: %{x:.2f}<br>φ: %{y:.2f}<extra></extra>",
            )
        )
        
        # Agregar barras de dispersión en gris (sin labels)
        best_pred = None
        if len(results) > 0:
            try:
                best_pred = predict(results[0], x)
            except ValueError:
                if results[0].model_type == "sqrt_fines" and fines is not None:
                    params = results[0].params
                    best_pred = params["a"] * np.sqrt(x) + params["b"] * fines + params["c"]
                else:
                    best_pred = None
            if best_pred is not None:
                line_x = []
                line_y = []
                for xi, yi, ypi in zip(x, y, best_pred):
                    line_x.extend([xi, xi, None])
                    line_y.extend([yi, ypi, None])
                fig.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode="lines",
                        name="Dispersión",
                        line={"color": "#5b5b5b", "width": 1.8, "dash": "dot"},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )


    fines_mean = None
    if fines is not None and len(fines) > 0:
        fines_mean = float(np.mean(fines))

    for res in results:
        if res.model_type == "sqrt_fines":
            if fines_mean is None:
                continue
            y_line = res.params["a"] * np.sqrt(x_line) + res.params["b"] * fines_mean + res.params["c"]
            trace_name = f"{res.name} (FC={fines_mean:.1f})"
        else:
            y_line = predict(res, x_line)
            trace_name = res.name
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=trace_name,
                line={"color": "#8b1e3f", "width": 3},
                hoverinfo="skip",
            )
        )

    title_band_color = "#1f4e79"
    title_text_color = "#ffffff"
    annotations = []
    if title and not embed_mode:
        annotations.append(
            {
                "x": 0.5,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "text": title,
                "showarrow": False,
                "align": "center",
                "yanchor": "middle",
                "font": {
                    "family": "Times New Roman, Georgia, serif",
                    "size": 22,
                    "color": title_text_color,
                },
            }
        )

    stats_lines = []
    if equation_text:
        stats_lines.append(equation_text)
    if r2_value is not None:
        stats_lines.append(f"R² = {r2_value:.4f}")
    if rmse_value is not None:
        stats_lines.append(f"RMSE = {rmse_value:.4f}")
    if mape_value is not None:
        stats_lines.append(f"Dispersión = {mape_value:.2f}%")
    if stats_lines and not embed_mode:
        annotations.append(
            {
                "x": 0.96,
                "y": 0.93,
                "xref": "paper",
                "yref": "paper",
                "text": "<br>".join(stats_lines),
                "showarrow": False,
                "align": "center",
                "bgcolor": "rgba(255,255,255,0.9)",
                "bordercolor": "#2f3b52",
                "borderwidth": 1.2,
                "borderpad": 8,
                "font": {
                    "size": 13,
                    "color": "#1b1b1b",
                    "family": "Times New Roman, Georgia, serif",
                },
            }
        )

    # Build shapes list (avoid nested lists which Plotly rejects)
    if embed_mode:
        shapes_list = []
    else:
        if title:
            shapes_list = [
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.0,
                    "x1": 1.0,
                    "y0": 1.02,
                    "y1": 1.14,
                    "fillcolor": title_band_color,
                    "line": {"width": 0},
                },
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.76,
                    "x1": 1.0,
                    "y0": 0.0,
                    "y1": 1.0,
                    "fillcolor": "rgba(245,247,250,0.95)",
                    "layer": "below",
                    "line": {"color": "#c7c7c7", "width": 1},
                },
            ]
        else:
            shapes_list = [
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.76,
                    "x1": 1.0,
                    "y0": 0.0,
                    "y1": 1.0,
                    "fillcolor": "rgba(245,247,250,0.95)",
                    "layer": "below",
                    "line": {"color": "#c7c7c7", "width": 1},
                }
            ]

    # Build shapes list (avoid nested lists which Plotly rejects)
    if embed_mode:
        shapes_list = []
    else:
        if title:
            shapes_list = [
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.0,
                    "x1": 1.0,
                    "y0": 1.02,
                    "y1": 1.14,
                    "fillcolor": title_band_color,
                    "line": {"width": 0},
                },
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.76,
                    "x1": 1.0,
                    "y0": 0.0,
                    "y1": 1.0,
                    "fillcolor": "rgba(245,247,250,0.95)",
                    "layer": "below",
                    "line": {"color": "#c7c7c7", "width": 1},
                },
            ]
        else:
            shapes_list = [
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.76,
                    "x1": 1.0,
                    "y0": 0.0,
                    "y1": 1.0,
                    "fillcolor": "rgba(245,247,250,0.95)",
                    "layer": "below",
                    "line": {"color": "#c7c7c7", "width": 1},
                }
            ]

    # Build shapes list (avoid nested lists which Plotly rejects)
    if embed_mode:
        shapes_list = []
    else:
        if title:
            shapes_list = [
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.0,
                    "x1": 1.0,
                    "y0": 1.02,
                    "y1": 1.14,
                    "fillcolor": title_band_color,
                    "line": {"width": 0},
                },
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.76,
                    "x1": 1.0,
                    "y0": 0.0,
                    "y1": 1.0,
                    "fillcolor": "rgba(245,247,250,0.95)",
                    "layer": "below",
                    "line": {"color": "#c7c7c7", "width": 1},
                },
            ]
        else:
            shapes_list = []

    fig.update_layout(
        title={"text": ""},
        xaxis={
            "title": {
                "text": xlabel,
                "font": {
                    "family": "Times New Roman, Georgia, serif",
                    "size": 18,
                    "color": "#111111",
                },
            },
            "domain": [0.0, 1.0] if embed_mode else [0.0, 0.72],
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "#e7e7e7",
            "tickfont": {"family": "Times New Roman, Georgia, serif", "size": 13},
            "zeroline": False,
            "showline": True,
            "linecolor": "#2f3b52",
            "linewidth": 1.6,
            "mirror": True,
        },
        yaxis={
            "title": {
                "text": ylabel,
                "font": {
                    "family": "Times New Roman, Georgia, serif",
                    "size": 18,
                    "color": "#111111",
                },
            },
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "#e7e7e7",
            "tickfont": {"family": "Times New Roman, Georgia, serif", "size": 13},
            "zeroline": False,
            "showline": True,
            "linecolor": "#2f3b52",
            "linewidth": 1.6,
            "mirror": True,
        },
        template="plotly_white",
        font={"family": "Times New Roman, Georgia, serif", "size": 12, "color": "#111111"},
        showlegend=not has_labels,
        legend={
            "orientation": "h",
            "x": 0.99,
            "xanchor": "right",
            "y": 0.06,
            "yanchor": "bottom",
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "#c7c7c7",
            "borderwidth": 1,
            "font": {"family": "Times New Roman, Georgia, serif", "size": 15},
        },
        annotations=annotations,
        shapes=shapes_list,
        margin={"l": 90, "r": 40, "t": 160 if not embed_mode else 30, "b": 80 if not embed_mode else 40},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    # Cuando se genera HTML para embed (ej. Power BI) no fijar la altura
    if not embed_mode:
        fig.update_layout(height=640)

    if prefix_counts and not embed_mode:
        values = [prefix_counts[p] for p in (prefix_order or prefix_counts.keys())]
        total_points = sum(values)
        text_positions = [
            "outside" if (value / total_points * 100) < 6 else "inside"
            for value in values
        ]
        fig.add_trace(
            go.Pie(
                labels=prefix_order or list(prefix_counts.keys()),
                values=values,
                hole=0.25,
                textinfo="label+percent",
                textposition=text_positions,
                marker={
                    "colors": [prefix_color_map[p] for p in (prefix_order or prefix_counts.keys())],
                    "line": {"color": "#2f3b52", "width": 1},
                }
                if prefix_color_map
                else None,
                domain={"x": [0.80, 0.98], "y": [0.12, 0.55]},
                sort=False,
            )
        )
        fig.add_annotation(
            x=0.95,
            y=0.58,
            xref="paper",
            yref="paper",
            text=f"Total de puntos: {total_points}",
            showarrow=False,
            align="center",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            font={"family": "Times New Roman, Georgia, serif", "size": 12, "color": "#111111"},
        )

    return to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "displaylogo": False,
            "editable": False,
            "scrollZoom": False,
            "responsive": True,
        },
    )


def plot_model_comparison_3d(
    x,
    fines,
    y,
    result,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str | None = None,
    equation_text: str | None = None,
    r2_value: float | None = None,
    rmse_value: float | None = None,
    mape_value: float | None = None,
    point_labels: np.ndarray | None = None,
    embed_mode: bool = False,
):
    x = np.asarray(x, dtype=float)
    fines = np.asarray(fines, dtype=float)
    y = np.asarray(y, dtype=float)

    x_grid = np.linspace(np.min(x), np.max(x), 40)
    fines_grid = np.linspace(np.min(fines), np.max(fines), 40)
    xx, ff = np.meshgrid(x_grid, fines_grid)

    a = result.params["a"]
    b = result.params["b"]
    c = result.params["c"]
    zz = a * np.sqrt(xx) + b * ff + c

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xx,
            y=ff,
            z=zz,
            colorscale=[
                [0.0, "#2b6ea6"],
                [0.5, "#3f86bf"],
                [1.0, "#76b5e6"],
            ],
            opacity=0.82,
            showscale=False,
            name="Superficie",
            hovertemplate="N60: %{x:.2f}<br>Finos (FC, %): %{y:.2f}<br>φ: %{z:.2f}<extra>Superficie</extra>",
        )
    )
    has_labels = point_labels is not None and len(point_labels) > 0 and len(point_labels) == len(x)
    prefix_counts = None
    prefix_order = None
    prefix_color_map = None
    if has_labels:
        prefix_dict = {}
        for i, label in enumerate(point_labels):
            label_str = str(label).strip()
            match = re.match(r"([A-Z]+)", label_str)
            prefix = match.group(1) if match else "OTRO"
            if prefix not in prefix_dict:
                prefix_dict[prefix] = {"indices": [], "labels": []}
            prefix_dict[prefix]["indices"].append(i)
            prefix_dict[prefix]["labels"].append(label_str)

        prefix_counts = {prefix: len(data["indices"]) for prefix, data in prefix_dict.items()}
        prefix_order = list(prefix_dict.keys())

        color_palette = [
            "#1f4e79",
            "#c00000",
            "#70ad47",
            "#ffc000",
            "#5b9bd5",
            "#ed7d31",
            "#a5a5a5",
            "#44546a",
            "#8b1e3f",
            "#2f3b52",
        ]

        prefix_color_map = {}
        for idx, (prefix, data) in enumerate(prefix_dict.items()):
            indices = np.array(data["indices"])
            labels = data["labels"]
            color = color_palette[idx % len(color_palette)]
            prefix_color_map[prefix] = color
            fig.add_trace(
                go.Scatter3d(
                    x=x[indices],
                    y=fines[indices],
                    z=y[indices],
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    textfont={"size": 9, "color": color},
                    name=prefix,
                    marker={"size": 4, "color": color, "symbol": "circle", "line": {"color": "#ffffff", "width": 1}},
                    hovertemplate="%{text}<br>N60: %{x:.2f}<br>Finos (FC, %): %{y:.2f}<br>φ: %{z:.2f}<extra></extra>",
                )
            )
        
        # Agregar barras de dispersión por grupo con el color del grupo
        params = result.params
        z_pred = params["a"] * np.sqrt(x) + params["b"] * fines + params["c"]
        for prefix, data in prefix_dict.items():
            indices = np.array(data["indices"])
            color = prefix_color_map[prefix]
            seg_x = []
            seg_y = []
            seg_z = []
            for i in indices:
                seg_x.extend([x[i], x[i], None])
                seg_y.extend([fines[i], fines[i], None])
                seg_z.extend([y[i], z_pred[i], None])
            fig.add_trace(
                go.Scatter3d(
                    x=seg_x,
                    y=seg_y,
                    z=seg_z,
                    mode="lines",
                    line={"color": color, "width": 3.5, "dash": "dot"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    else:
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=fines,
                z=y,
                mode="markers",
                name="Datos",
                marker={"size": 4, "color": "#8b1e3f", "symbol": "circle", "line": {"color": "#ffffff", "width": 1}},
                hovertemplate="N60: %{x:.2f}<br>Finos (FC, %): %{y:.2f}<br>φ: %{z:.2f}<extra>Datos</extra>",
            )
        )
        
        # Agregar barras de dispersión en gris (sin labels)
        params = result.params
        z_pred = params["a"] * np.sqrt(x) + params["b"] * fines + params["c"]
        if len(x) > 0:
            seg_x = []
            seg_y = []
            seg_z = []
            for xi, fi, yi, zi in zip(x, fines, y, z_pred):
                seg_x.extend([xi, xi, None])
                seg_y.extend([fi, fi, None])
                seg_z.extend([yi, zi, None])
            fig.add_trace(
                go.Scatter3d(
                    x=seg_x,
                    y=seg_y,
                    z=seg_z,
                    mode="lines",
                    line={"color": "#5b5b5b", "width": 3.5, "dash": "dot"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )


    title_band_color = "#1f4e79"
    title_text_color = "#ffffff"
    annotations = []
    if title and not embed_mode:
        annotations.append(
            {
                "x": 0.5,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "text": title,
                "showarrow": False,
                "align": "center",
                "yanchor": "middle",
                "font": {
                    "family": "Times New Roman, Georgia, serif",
                    "size": 22,
                    "color": title_text_color,
                },
            }
        )

    if equation_text is None:
        equation_text = result.equation

    stats_lines = []
    if equation_text:
        split_equation = equation_text.replace(
            "*sqrt(N<sub>60</sub>) ", "*sqrt(N<sub>60</sub>)<br>", 1
        )
        stats_lines.append(f"<b><span style='color:#1f4e79'>{split_equation}</span></b>")
    if r2_value is not None:
        stats_lines.append(f"R² = {r2_value:.4f}")
    if rmse_value is not None:
        stats_lines.append(f"RMSE = {rmse_value:.4f}")
    if mape_value is not None:
        stats_lines.append(f"Dispersión = {mape_value:.2f}%")
    if stats_lines and not embed_mode:
        annotations.append(
            {
                "x": 0.96,
                "y": 0.93,
                "xref": "paper",
                "yref": "paper",
                "text": "<br>".join(stats_lines),
                "showarrow": False,
                "align": "center",
                "bgcolor": "rgba(255,255,255,0.9)",
                "bordercolor": "#2f3b52",
                "borderwidth": 1.2,
                "borderpad": 8,
                "font": {
                    "size": 13,
                    "color": "#1b1b1b",
                    "family": "Times New Roman, Georgia, serif",
                },
            }
        )

    fig.update_layout(
        title={"text": ""},
        scene={
            "xaxis": {
                "title": xlabel,
                "titlefont": {"size": 14},
                "tickfont": {"size": 11, "color": "#111111"},
                "showline": True,
                "linecolor": "#1f2a3a",
                "linewidth": 2,
                "gridcolor": "#9aa7b4",
                "gridwidth": 2,
                "zeroline": False,
            },
            "yaxis": {
                "title": ylabel,
                "titlefont": {"size": 14},
                "tickfont": {"size": 11, "color": "#111111"},
                "showline": True,
                "linecolor": "#1f2a3a",
                "linewidth": 2,
                "gridcolor": "#9aa7b4",
                "gridwidth": 2,
                "zeroline": False,
            },
            "zaxis": {
                "title": zlabel,
                "titlefont": {"size": 14},
                "tickfont": {"size": 11, "color": "#111111"},
                "showline": True,
                "linecolor": "#1f2a3a",
                "linewidth": 2,
                "gridcolor": "#9aa7b4",
                "gridwidth": 2,
                "zeroline": False,
            },
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]} if embed_mode else {"x": [0.0, 0.72], "y": [0.0, 1.0]},
            "aspectmode": "auto",
        },
        scene_dragmode="orbit",
        margin={"l": 20, "r": 20, "t": 20, "b": 20} if embed_mode else {"l": 90, "r": 40, "t": 160, "b": 80},
        autosize=True,
        height=640 if not embed_mode else None,
        template="plotly_white",
        font={"family": "Times New Roman, Georgia, serif", "size": 12, "color": "#111111"},
        showlegend=False,
        annotations=annotations,
        shapes=shapes_list,
    )

    if prefix_counts and not embed_mode:
        values = [prefix_counts[p] for p in (prefix_order or prefix_counts.keys())]
        total_points = sum(values)
        text_positions = [
            "outside" if (value / total_points * 100) < 6 else "inside"
            for value in values
        ]
        fig.add_trace(
            go.Pie(
                labels=prefix_order or list(prefix_counts.keys()),
                values=values,
                hole=0.25,
                textinfo="label+percent",
                textposition=text_positions,
                marker={
                    "colors": [prefix_color_map[p] for p in (prefix_order or prefix_counts.keys())],
                    "line": {"color": "#2f3b52", "width": 1},
                }
                if prefix_color_map
                else None,
                domain={"x": [0.80, 0.98], "y": [0.12, 0.55]},
                sort=False,
            )
        )
        fig.add_annotation(
            x=0.95,
            y=0.58,
            xref="paper",
            yref="paper",
            text=f"Total de puntos: {total_points}",
            showarrow=False,
            align="center",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            font={"family": "Times New Roman, Georgia, serif", "size": 12, "color": "#111111"},
        )

    return to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "displaylogo": False,
            "editable": False,
            "scrollZoom": True,
            "displayModeBar": False,
            "responsive": True,
        },
    )


def plot_author_comparison(
    x,
    your_y,
    your_column_name,
    comparison_series,
    field_points_x,
    field_points_y,
    x_label,
    ylabel,
    title,
    target_type="phi",
    r2_by_series=None,
    r2_your=None,
    fines_low=None,
    fines_high=None,
    fines_low_label=None,
    fines_high_label=None,
    show_fines_band=False,
    embed_mode=False,
):
    """
    Genera grafica de comparacion de correlaciones con colores distintos.
    
    Args:
        x: numpy array con valores de N60
        your_y: numpy array con tu correlacion
        your_column_name: nombre de tu columna
        comparison_series: dict con {nombre_autor: valores}
        x_label: etiqueta para eje X
        ylabel: etiqueta para eje Y (con unidades)
        title: titulo del grafico
        target_type: 'phi'
    """
    fig = go.Figure()
    
    # Paleta de colores y estilos de linea distintos para cada serie
    palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]
    dash_styles = [
        "solid",
        "dash",
        "dot",
        "dashdot",
        "longdash",
        "longdashdot",
        "dash",
        "dot",
        "longdash",
        "dashdot",
    ]
    
    def _wrap_legend_label(text, max_len=30):
        if not text:
            return text
        base_text = text
        r2_text = ""
        if " (R²" in text:
            base_text, r2_text = text.split(" (R²", 1)
            r2_text = "(R²" + r2_text
        words = base_text.split()
        lines = []
        current = ""
        for word in words:
            if not current:
                current = word
                continue
            if len(current) + len(word) + 1 <= max_len:
                current = f"{current} {word}"
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        if r2_text:
            lines.append(r2_text)
        return "<br>".join(lines)

    def _format_legend_name(label):
        if not label:
            return label
        clean = re.sub(r"\s*\(\d{4}\)$", "", label).strip()
        return clean

    def _legend_label(text, r2_text=""):
        if embed_mode:
            return _format_legend_name(text)
        return _wrap_legend_label(f"{text}{r2_text}")

    # Agregar series de comparacion (autores) con colores distintos
    color_idx = 0
    for col_name, col_values in comparison_series.items():
        if col_name != your_column_name and len(col_values) > 0:
            r2_text = ""
            if r2_by_series and col_name in r2_by_series and not embed_mode:
                r2_text = f" (R²={r2_by_series[col_name]:.4f})"
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=col_values,
                    mode="lines+markers",
                    name=_legend_label(col_name, r2_text),
                    line={
                        "color": palette[color_idx % len(palette)],
                        "width": 2,
                        "dash": dash_styles[color_idx % len(dash_styles)],
                    },
                    marker={"size": 7, "symbol": "circle"},
                )
            )
            color_idx += 1

    # Agregar banda de finos si aplica
    def _format_fines_label(value):
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if 0 <= numeric <= 1:
            return f"{numeric * 100:.0f}%"
        return f"{numeric:.0f}%" if numeric <= 100 else str(value)

    if show_fines_band and fines_low is not None and fines_high is not None:
        low_label = _format_fines_label(fines_low_label) or "Finos bajos"
        high_label = _format_fines_label(fines_high_label) or "Finos altos"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fines_low,
                mode="lines",
                name=low_label,
                line={"color": "#5b9bd5", "width": 2, "dash": "dot"},
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fines_high,
                mode="lines",
                name=high_label,
                fill="tonexty",
                fillcolor="rgba(91, 155, 213, 0.18)",
                line={"color": "#2b6ea6", "width": 2, "dash": "dash"},
                showlegend=False,
            )
        )

    # Agregar tu correlacion en ultimo lugar (resaltada)
    if len(your_y) > 0:
        your_r2_text = ""
        if r2_your is not None and not embed_mode:
            your_r2_text = f" (R²={r2_your:.4f})"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=your_y,
                mode="lines+markers",
                name=_legend_label(your_column_name, your_r2_text),
                line={"color": "#8b1e3f", "width": 2, "dash": "dash"},
                marker={"size": 7, "color": "#8b1e3f", "symbol": "circle", "line": {"color": "#ffffff", "width": 1}},
            )
        )

    # Agregar puntos de campo/ensayo
    if len(field_points_x) > 0 and len(field_points_y) > 0:
        fig.add_trace(
            go.Scatter(
                x=field_points_x,
                y=field_points_y,
                mode="markers",
                name="Puntos de ensayo",
                marker={"size": 12, "color": "#123A7D", "line": {"color": "#ffffff", "width": 1.5}, "symbol": "circle"},
            )
        )

    # Inicializar anotaciones y colores del encabezado (solo para la vista de comparación)
    title_band_color = "#1f4e79"
    title_text_color = "#ffffff"
    annotations = []

    if title and not embed_mode:
        annotations.append(
            {
                "x": 0.5,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "text": title,
                "showarrow": False,
                "align": "center",
                "yanchor": "middle",
                "font": {
                    "family": "Times New Roman, Georgia, serif",
                    "size": 22,
                    "color": title_text_color,
                },
            }
        )

    # Build shapes list (avoid nested lists which Plotly rejects)
    if embed_mode:
        shapes_list = []
    else:
        if title:
            shapes_list = [
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.0,
                    "x1": 1.0,
                    "y0": 1.02,
                    "y1": 1.14,
                    "fillcolor": title_band_color,
                    "line": {"width": 0},
                },
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.76,
                    "x1": 1.0,
                    "y0": 0.0,
                    "y1": 1.0,
                    "fillcolor": "rgba(245,247,250,0.95)",
                    "layer": "below",
                    "line": {"color": "#c7c7c7", "width": 1},
                },
            ]
        else:
            shapes_list = [
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0.76,
                    "x1": 1.0,
                    "y0": 0.0,
                    "y1": 1.0,
                    "fillcolor": "rgba(245,247,250,0.95)",
                    "layer": "below",
                    "line": {"color": "#c7c7c7", "width": 1},
                }
            ]

    fig.update_layout(
        title={"text": ""},
        xaxis_title=x_label or "Numero de golpes corregido, N<sub>60</sub>",
        yaxis_title=ylabel or "Valor",
        hovermode="x unified",
        template="plotly_white",
        height=600 if not embed_mode else None,
        showlegend=True,
        legend=(
            {
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.02,
                "yanchor": "bottom",
                "bgcolor": "rgba(255,255,255,0.96)",
                "bordercolor": "#4a4a4a",
                "borderwidth": 1.2,
                "tracegroupgap": 6,
                "font": {
                    "family": "Segoe UI, Arial, sans-serif",
                    "size": 14,
                    "color": "#111111",
                },
            }
            if embed_mode
            else {
                "x": 0.88,
                "y": 0.97,
                "xanchor": "center",
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.96)",
                "bordercolor": "#4a4a4a",
                "borderwidth": 1.2,
                "tracegroupgap": 6,
                "font": {
                    "family": "Times New Roman, Georgia, serif",
                    "size": 11,
                    "color": "#111111",
                },
            }
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Times New Roman, Georgia, serif", "size": 13, "color": "#111111"},
        xaxis={
            "title": {
                "font": {"family": "Times New Roman, Georgia, serif", "size": 18, "color": "#111111"}
            },
            "domain": [0.0, 1.0] if embed_mode else [0.0, 0.68],
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "#e0e0e0",
            "zeroline": False,
            "showline": True,
            "linecolor": "#2f3b52",
            "linewidth": 1.4,
            "tickfont": {"family": "Times New Roman, Georgia, serif", "size": 14, "color": "#111111"},
        },
        yaxis={
            "title": {
                "font": {"family": "Times New Roman, Georgia, serif", "size": 18, "color": "#111111"}
            },
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "#e0e0e0",
            "zeroline": False,
            "showline": True,
            "linecolor": "#2f3b52",
            "linewidth": 1.4,
            "tickfont": {"family": "Times New Roman, Georgia, serif", "size": 14, "color": "#111111"},
        },
        annotations=annotations,
        shapes=shapes_list,
        margin={"l": 90, "r": 40, "t": 160 if not embed_mode else 30, "b": 80 if not embed_mode else 40},
    )
    
    if show_fines_band and fines_low is not None and fines_high is not None:
        # Mostrar la banda de finos y sus indicadores también en embed_mode
        fig.add_annotation(
            x=0.88,
            y=0.24,
            xref="paper",
            yref="paper",
            text="<b>Finos</b>",
            showarrow=False,
            align="center",
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#4a4a4a",
            borderwidth=1.2,
            borderpad=6,
            font={"family": "Times New Roman, Georgia, serif", "size": 12, "color": "#111111"},
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.84,
            x1=0.92,
            y0=0.19,
            y1=0.19,
            line={"color": "#5b9bd5", "width": 2, "dash": "dot"},
        )
        fig.add_annotation(
            x=0.925,
            y=0.19,
            xref="paper",
            yref="paper",
            text=low_label,
            showarrow=False,
            align="left",
            font={"family": "Times New Roman, Georgia, serif", "size": 12, "color": "#111111"},
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.84,
            x1=0.92,
            y0=0.14,
            y1=0.14,
            line={"color": "#2b6ea6", "width": 2, "dash": "dash"},
        )
        fig.add_annotation(
            x=0.925,
            y=0.14,
            xref="paper",
            yref="paper",
            text=high_label,
            showarrow=False,
            align="left",
            font={"family": "Times New Roman, Georgia, serif", "size": 12, "color": "#111111"},
        )

    # Generate HTML. For embed_mode we inject a small CSS snippet to round
    # the legend box and force the border color/thickness (Power BI embed).
    html = fig.to_html(
        div_id="comparison-plot",
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displayModeBar": False,
            "displaylogo": False,
            "editable": False,
            "scrollZoom": False,
            "responsive": True,
        },
    )

    if embed_mode:
        style = (
            "<style>"
            "#comparison-plot .legend { background: white !important; border-radius: 8px !important; box-shadow: none !important; padding: 6px !important; border: 1px solid #123A7D !important; }"
            "#comparison-plot svg .legend rect { stroke: #123A7D !important; stroke-width: 1px !important; fill: white !important; }"
            "#comparison-plot svg .legend g rect { stroke: #123A7D !important; stroke-width: 1px !important; fill: white !important; }"
            "#comparison-plot .legend { color: #111111 !important; }"
            "</style>"
        )
        script = """
<script>
(function(){
    try{
        var root = document.getElementById('comparison-plot');
        if(!root) return;
        var svgs = root.getElementsByTagName('svg');
        for(var i=0;i<svgs.length;i++){
            var svg = svgs[i];
            // Find legend groups inside this SVG
            var legendGroups = svg.querySelectorAll('g[class*="legend"]');
            legendGroups.forEach(function(legend){
                var rects = Array.prototype.slice.call(legend.querySelectorAll('rect'));
                if(rects.length===0) return;
                // determine the largest rect (outer box)
                var outer = rects.reduce(function(best, r){ try{ var bb=r.getBBox(); var area=bb.width*bb.height; var bestBb=best?best.getBBox():{width:0,height:0}; var bestArea=bestBb.width*bestBb.height; return area>bestArea?r:best; }catch(e){return best;} }, null);
                if(outer){ try{ outer.setAttribute('fill','#ffffff'); outer.setAttribute('stroke','#123A7D'); outer.setAttribute('stroke-width','1'); outer.setAttribute('rx','8'); outer.setAttribute('ry','8'); }catch(e){} }
                try{ var outerBb = outer.getBBox(); }catch(e){ outerBb={width:Infinity,height:Infinity}; }
                // remove small rects that are clearly per-item boxes (width much smaller than outer)
                rects.forEach(function(r){ try{ if(r!==outer){ var bb=r.getBBox(); if(bb.width < outerBb.width*0.5){ r.parentNode.removeChild(r); } } }catch(e){} });
            });
        }
        var htmlLegends = root.querySelectorAll('.legend');
        htmlLegends.forEach(function(el){ el.style.background='white'; el.style.borderRadius='8px'; el.style.border='1px solid #123A7D'; el.style.padding='6px'; });
    }catch(e){console.warn(e);} 
})();
</script>
"""

        # Insert style into <head> if present, and script before </body> to run after render
        head_close = "</head>"
        idx_head = html.find(head_close)
        if idx_head != -1:
            html = html[:idx_head] + style + html[idx_head:]
        else:
            html = style + html

        body_close = "</body>"
        idx_body = html.find(body_close)
        if idx_body != -1:
            html = html[:idx_body] + script + html[idx_body:]
        else:
            html = html + script

    return html


def plot_fines_phi_relationship(
    fines,
    phi,
    fines_label="Porcentaje de finos (%)",
    phi_label="Ángulo de fricción interna, φ (°)",
    title="Relación entre finos y φ",
    regression_type="linear",
    phi_fit=None,
    point_labels=None,
    embed_mode=False,
):
    fines = np.asarray(fines, dtype=float)
    phi = np.asarray(phi, dtype=float)
    phi_fit = np.asarray(phi_fit, dtype=float) if phi_fit is not None else phi
    point_labels = np.asarray(point_labels, dtype=str) if point_labels is not None else None

    if len(phi_fit) != len(phi):
        raise ValueError("phi_fit debe tener la misma longitud que phi.")
    if point_labels is not None and len(point_labels) != len(phi):
        raise ValueError("point_labels debe tener la misma longitud que los datos.")

    valid_mask = np.isfinite(fines) & np.isfinite(phi) & np.isfinite(phi_fit)
    fines = fines[valid_mask]
    phi = phi[valid_mask]
    phi_fit = phi_fit[valid_mask]
    if point_labels is not None:
        point_labels = point_labels[valid_mask]

    fig = go.Figure()

    has_labels = (
        point_labels is not None
        and len(point_labels) > 0
        and len(point_labels) == len(fines)
    )
    if has_labels:
        prefix_dict = {}
        for i, label in enumerate(point_labels):
            label_str = str(label).strip()
            match = re.match(r"([A-Z]+)", label_str)
            prefix = match.group(1) if match else "OTRO"
            if prefix not in prefix_dict:
                prefix_dict[prefix] = {"indices": [], "labels": []}
            prefix_dict[prefix]["indices"].append(i)
            prefix_dict[prefix]["labels"].append(label_str)

        color_palette = [
            "#1f4e79",
            "#c00000",
            "#70ad47",
            "#ffc000",
            "#5b9bd5",
            "#ed7d31",
            "#a5a5a5",
            "#44546a",
            "#e2efda",
            "#fce4d6",
        ]

        for idx, (prefix, data) in enumerate(prefix_dict.items()):
            indices = np.array(data["indices"])
            labels = data["labels"]
            color = color_palette[idx % len(color_palette)]
            fig.add_trace(
                go.Scatter(
                    x=fines[indices],
                    y=phi[indices],
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    textfont={"size": 10, "color": color},
                    name=prefix,
                    marker={
                        "color": color,
                        "size": 9,
                        "line": {"color": "#ffffff", "width": 1.2},
                        "opacity": 0.95,
                    },
                    hovertemplate="%{text}<br>FC: %{x:.2f}<br>φ: %{y:.2f}<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=fines,
                y=phi,
                mode="markers",
                name="Puntos de datos",
                marker={
                    "color": "#1f4e79",
                    "size": 11,
                    "symbol": "circle",
                    "line": {"color": "#ffffff", "width": 1.5},
                    "opacity": 0.95,
                },
                hovertemplate=f"{fines_label}: %{{x:.2f}}<br>{phi_label}: %{{y:.2f}}<extra></extra>",
            )
        )

    equation_text = None
    r2_value = None
    rmse_value = None
    if len(fines) >= 2:
        if regression_type == "logarithmic":
            positive_mask = fines > 0
            if np.sum(positive_mask) < 2:
                raise ValueError(
                    "No hay suficientes valores de finos mayores a cero para regresión logarítmica."
                )
            log_fines = np.log(fines[positive_mask])
            fit_phi = phi_fit[positive_mask]
            slope, intercept = np.polyfit(log_fines, fit_phi, 1)
            line_x = np.linspace(np.min(fines[positive_mask]), np.max(fines[positive_mask]), 200)
            line_y = slope * np.log(line_x) + intercept
            y_pred = slope * np.log(fines[positive_mask]) + intercept
            r2_value = r2_score(fit_phi, y_pred)
            rmse_value = rmse(fit_phi, y_pred)
            equation_text = f"φ = {slope:.4f}·ln(FC) + {intercept:.4f}"
            trace_name = "Ajuste logarítmico"
        else:
            slope, intercept = np.polyfit(fines, phi_fit, 1)
            line_x = np.linspace(np.min(fines), np.max(fines), 200)
            line_y = slope * line_x + intercept
            y_pred = slope * fines + intercept
            r2_value = r2_score(phi_fit, y_pred)
            rmse_value = rmse(phi_fit, y_pred)
            equation_text = f"φ = {slope:.4f}·FC + {intercept:.4f}"
            trace_name = "Ajuste lineal"
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name=trace_name,
                line={"color": "#8b1e3f", "width": 3.2},
                hoverinfo="skip",
            )
        )

    title_band_color = "#1f4e79"
    title_text_color = "#ffffff"
    annotations = []
    if title and not embed_mode:
        annotations.append(
            {
                "x": 0.5,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "text": title,
                "showarrow": False,
                "align": "center",
                "yanchor": "middle",
                "font": {
                    "family": "Times New Roman, Georgia, serif",
                    "size": 22,
                    "color": title_text_color,
                },
            }
        )

    stats_lines = []
    if equation_text:
        stats_lines.append(f"<b><span style='color:#1f4e79'>{equation_text}</span></b>")
    if r2_value is not None:
        stats_lines.append(f"R² = {r2_value:.4f}")
    if rmse_value is not None:
        stats_lines.append(f"RMSE = {rmse_value:.4f}")
    if stats_lines and not embed_mode:
        annotations.append(
            {
                "x": 0.96,
                "y": 0.93,
                "xref": "paper",
                "yref": "paper",
                "text": "<br>".join(stats_lines),
                "showarrow": False,
                "align": "center",
                "bgcolor": "rgba(255,255,255,0.9)",
                "bordercolor": "#2f3b52",
                "borderwidth": 1.2,
                "borderpad": 8,
                "font": {
                    "size": 13,
                    "color": "#1b1b1b",
                    "family": "Times New Roman, Georgia, serif",
                },
            }
        )

    fig.update_layout(
        title={"text": ""},
        xaxis={
            "title": {
                "text": fines_label,
                "font": {
                    "family": "Times New Roman, Georgia, serif",
                    "size": 18,
                    "color": "#111111",
                },
            },
            "domain": [0.0, 1.0] if embed_mode else [0.0, 0.72],
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "#e7e7e7",
            "tickfont": {"family": "Times New Roman, Georgia, serif", "size": 13},
            "zeroline": False,
            "showline": True,
            "linecolor": "#2f3b52",
            "linewidth": 1.6,
            "mirror": True,
        },
        yaxis={
            "title": {
                "text": phi_label,
                "font": {
                    "family": "Times New Roman, Georgia, serif",
                    "size": 18,
                    "color": "#111111",
                },
            },
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "#e7e7e7",
            "tickfont": {"family": "Times New Roman, Georgia, serif", "size": 13},
            "zeroline": False,
            "showline": True,
            "linecolor": "#2f3b52",
            "linewidth": 1.6,
            "mirror": True,
        },
        template="plotly_white",
        font={"family": "Times New Roman, Georgia, serif", "size": 12, "color": "#111111"},
        showlegend=False if embed_mode else True,
        legend={
            "orientation": "h",
            "x": 0.99,
            "xanchor": "right",
            "y": 0.06,
            "yanchor": "bottom",
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "#c7c7c7",
            "borderwidth": 1,
            "font": {"family": "Times New Roman, Georgia, serif", "size": 15},
        },
        annotations=annotations,
        shapes=[] if embed_mode else ([{"type": "rect", "xref": "paper", "yref": "paper", "x0": 0.0, "x1": 1.0, "y0": 1.02, "y1": 1.14, "fillcolor": title_band_color, "line": {"width": 0}}, {"type": "rect", "xref": "paper", "yref": "paper", "x0": 0.76, "x1": 1.0, "y0": 0.0, "y1": 1.0, "fillcolor": "rgba(245,247,250,0.95)", "layer": "below", "line": {"color": "#c7c7c7", "width": 1}}] if title else [{"type": "rect", "xref": "paper", "yref": "paper", "x0": 0.76, "x1": 1.0, "y0": 0.0, "y1": 1.0, "fillcolor": "rgba(245,247,250,0.95)", "layer": "below", "line": {"color": "#c7c7c7", "width": 1}}]),
        margin={"l": 90, "r": 40, "t": 160 if not embed_mode else 30, "b": 80 if not embed_mode else 40},
        height=640 if not embed_mode else None,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig.to_html(
        div_id="fines-phi-plot",
        include_plotlyjs="cdn",
        config={
            "displayModeBar": False,
            "displaylogo": False,
            "editable": False,
            "scrollZoom": False,
            "responsive": True,
        },
    )


def save_fig(fig, path: Path) -> None:
    html = to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "editable": False,
            "scrollZoom": False,
            "responsive": True,
        },
    )
    path.write_text(html, encoding="utf-8")
