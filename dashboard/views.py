import base64
from pathlib import Path

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render

from models.regression import analyze_target
from utils.io import (
    get_excel_sheet_names_from_bytes,
    load_all_dataframes_from_bytes,
    load_dataframe_from_bytes,
    resolve_sheet_query,
)
from utils.plotting import (
    plot_author_comparison,
    plot_fines_phi_relationship,
    plot_model_comparison,
    plot_model_comparison_3d,
)

from .forms import FinesUploadForm, UploadForm
from .persistence import (
    load_latest_upload_metadata,
    load_plot_html,
    load_plot_html_embed,
    load_plot_metadata,
    load_snapshot_status,
    load_uploaded_file,
    save_plot_snapshot,
    save_uploaded_file,
)


UPLOAD_SESSION_KEY = "dashboard_uploaded_file"
UPLOAD_NAME_SESSION_KEY = "dashboard_uploaded_file_name"
UPLOAD_SHEET_SESSION_KEY = "dashboard_uploaded_sheet"
PREVIEW_ROWS = 8
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
LATEST_POWERBI_PLOT_PATH = OUTPUTS_DIR / "latest_powerbi_plot.html"


def _base_sections():
    return {
        "phi": {
            "target": "phi",
            "heading": "Estimación del ángulo de fricción φ",
            "equation_title": "Ecuación",
            "equation_placeholder": "Aún no hay ecuación calculada",
            "graph_title": "Curva de ajuste",
            "graph_placeholder_title": "Aún no hay datos para mostrar",
            "graph_placeholder_text": "Sube un archivo y selecciona las columnas N<sub>60</sub> y φ para generar la regresión del ángulo de fricción.",
            "graph_button_label": "Previsualizar archivo",
            "plot_html": None,
            "best": None,
        },
    }


def _encode_file_bytes(file_bytes):
    return base64.b64encode(file_bytes).decode("ascii")


def _decode_file_bytes(encoded_file):
    return base64.b64decode(encoded_file.encode("ascii"))


def _store_uploaded_file(request, uploaded_file, sheet_name=None):
    file_bytes = uploaded_file.read()
    request.session[UPLOAD_SESSION_KEY] = _encode_file_bytes(file_bytes)
    request.session[UPLOAD_NAME_SESSION_KEY] = uploaded_file.name
    request.session.modified = True
    sheet_names = get_excel_sheet_names_from_bytes(file_bytes, uploaded_file.name)
    resolved_sheet = sheet_name
    if resolved_sheet is None and sheet_names:
        resolved_sheet = sheet_names[0]
    if resolved_sheet:
        request.session[UPLOAD_SHEET_SESSION_KEY] = resolved_sheet
    else:
        request.session.pop(UPLOAD_SHEET_SESSION_KEY, None)
    save_uploaded_file(file_bytes, uploaded_file.name, resolved_sheet)
    df = load_dataframe_from_bytes(file_bytes, uploaded_file.name, sheet_name=resolved_sheet)
    return df, sheet_names, resolved_sheet


def _load_session_dataframe(request, sheet_name=None):
    encoded_file = request.session.get(UPLOAD_SESSION_KEY)
    filename = request.session.get(UPLOAD_NAME_SESSION_KEY)
    if not encoded_file or not filename:
        file_bytes, filename, stored_sheet = load_uploaded_file()
        if not file_bytes or not filename:
            raise ValueError("Primero sube un archivo CSV o Excel.")
        request.session[UPLOAD_SESSION_KEY] = _encode_file_bytes(file_bytes)
        request.session[UPLOAD_NAME_SESSION_KEY] = filename
        if stored_sheet:
            request.session[UPLOAD_SHEET_SESSION_KEY] = stored_sheet
        request.session.modified = True
    else:
        file_bytes = _decode_file_bytes(encoded_file)
        stored_sheet = request.session.get(UPLOAD_SHEET_SESSION_KEY)
    sheet_names = get_excel_sheet_names_from_bytes(file_bytes, filename)
    resolved_sheet = sheet_name or request.session.get(UPLOAD_SHEET_SESSION_KEY) or stored_sheet
    if resolved_sheet is None and sheet_names:
        resolved_sheet = sheet_names[0]
    if resolved_sheet:
        request.session[UPLOAD_SHEET_SESSION_KEY] = resolved_sheet
    df = load_dataframe_from_bytes(file_bytes, filename, sheet_name=resolved_sheet)
    return df, sheet_names, resolved_sheet


def _build_preview_context(df, post_data=None, sheet_names=None, selected_sheet=None):
    columns = list(df.columns)
    default_n60 = "N60" if "N60" in columns else ""
    default_phi = "phi" if "phi" in columns else ""
    default_point_code = ""
    default_fines = _guess_fines_column(columns) or ""
    excluded_columns = []
    sheet_options = sheet_names or []
    selected_sheet_value = selected_sheet or (sheet_options[0] if sheet_options else "")

    if post_data is not None:
        default_n60 = post_data.get("selected_n60", default_n60)
        default_phi = post_data.get("selected_phi", default_phi)
        default_point_code = post_data.get("selected_point_code", default_point_code)
        default_fines = post_data.get("selected_fines", default_fines)
        excluded_columns = post_data.getlist("exclude_columns")
        selected_sheet_value = post_data.get("selected_sheet", selected_sheet_value)
    use_fixed_c = False
    fixed_c_value = ""
    if post_data is not None:
        use_fixed_c = post_data.get("use_fixed_c") == "on"
        fixed_c_value = post_data.get("fixed_c_value", "")

    default_model_type = "linear"
    if post_data is not None:
        default_model_type = post_data.get("model_type", default_model_type)

    return {
        "has_preview": True,
        "preview_headers": columns,
        "preview_rows": [
            [row.get(column) for column in columns]
            for row in df.head(PREVIEW_ROWS).to_dict(orient="records")
        ],
        "selected_n60": default_n60,
        "selected_phi": default_phi,
        "selected_point_code": default_point_code,
        "selected_fines": default_fines,
        "selected_model_type": default_model_type,
        "use_fixed_c": use_fixed_c,
        "fixed_c_value": fixed_c_value,
        "excluded_columns": excluded_columns,
        "sheet_options": sheet_options,
        "selected_sheet": selected_sheet_value,
    }


def _guess_fines_column(columns):
    for col in columns:
        raw_key = str(col).strip().lower()
        key = "".join(ch for ch in raw_key if ch.isalnum())
        if key in {
            "fc",
            "fines",
            "finos",
            "finospct",
            "finespct",
            "porcentajefinos",
            "porcentajedefinos",
            "finespercent",
            "percentfines",
            "pctfinos",
            "pfinos",
        }:
            return col
        if "fino" in key or "fines" in key:
            return col
    return None


def _apply_column_exclusions(df, excluded_columns):
    if not excluded_columns:
        return df
    columns_to_drop = [col for col in excluded_columns if col in df.columns]
    if not columns_to_drop:
        return df
    return df.drop(columns=columns_to_drop)


def _save_latest_plot_html(plot_html, metadata=None, plot_html_embed=None):
    save_plot_snapshot(plot_html, metadata, plot_html_embed=plot_html_embed)


def _hydrate_context_from_snapshot(request, context):
    """Carga el último Excel y gráfica guardados en el servidor (todos los dispositivos)."""
    if not context.get("has_preview"):
        try:
            df, sheet_names, selected_sheet = _load_session_dataframe(request)
            context.update(
                _build_preview_context(
                    df,
                    sheet_names=sheet_names,
                    selected_sheet=selected_sheet,
                )
            )
            context["has_preview"] = True
        except ValueError:
            pass

    plot_html = load_plot_html()
    metadata = load_plot_metadata()
    if not plot_html:
        return context

    view_mode = context.get("view_mode")
    if metadata.get("view_mode") != view_mode:
        return context

    if view_mode == "regression":
        has_plot = any(section.get("plot_html") for section in context.get("sections", []))
        if has_plot:
            return context
        for section in context.get("sections", []):
            if section.get("target") != "phi":
                continue
            section["plot_html"] = plot_html
            for key in ("equation", "r2", "rmse", "mape", "model_name"):
                if key in metadata:
                    section[key] = metadata[key]
    elif view_mode == "comparison" and not context.get("comparison_plot_html"):
        context["comparison_plot_html"] = plot_html
        for key in ("your_column", "r2_by_series", "r2_your"):
            if key in metadata:
                context[key] = metadata[key]
    elif view_mode == "fines" and not context.get("fines_plot_html"):
        context["fines_plot_html"] = plot_html
        for key in ("selected_fines", "selected_phi"):
            if key in metadata:
                context[key] = metadata[key]

    context["loaded_from_shared_snapshot"] = True
    context["shared_snapshot_at"] = load_snapshot_status().get("updated_at")
    return context


def _load_latest_plot_html():
    plot_html = load_plot_html()
    if plot_html:
        return plot_html
    if LATEST_POWERBI_PLOT_PATH.exists():
        return LATEST_POWERBI_PLOT_PATH.read_text(encoding="utf-8")
    return None


def _parse_fixed_c(post_data):
    use_fixed_c = post_data.get("use_fixed_c") == "on"
    if not use_fixed_c:
        return None
    value = post_data.get("fixed_c_value", "").strip()
    if value == "":
        raise ValueError("Debes ingresar una constante cuando activas esta opcion.")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError("La constante debe ser un numero valido.") from exc


def _train_one_target(
    df,
    x_column,
    target_column,
    target_name,
    model_type,
    point_code_column=None,
    fines_column=None,
    fixed_c=None,
):
    if x_column not in df.columns:
        raise ValueError("La columna seleccionada para N<sub>60</sub> no existe en el archivo.")
    if target_column not in df.columns:
        raise ValueError(f"La columna seleccionada para {target_name} no existe en el archivo.")

    columns_to_keep = [x_column, target_column]
    if model_type == "sqrt_fines":
        if not fines_column:
            raise ValueError("Debes seleccionar la columna de finos para este modelo.")
        if fines_column not in df.columns:
            raise ValueError("La columna seleccionada para finos no existe en el archivo.")
        columns_to_keep.append(fines_column)
    if point_code_column and point_code_column in df.columns:
        columns_to_keep.append(point_code_column)
    
    subset = df[columns_to_keep].copy()
    rename_columns = ["N60", target_name]
    if model_type == "sqrt_fines":
        rename_columns.append("fines")
    if point_code_column and point_code_column in df.columns:
        rename_columns.append("point_code")
    subset.columns = rename_columns
    subset["N60"] = pd.to_numeric(subset["N60"], errors="coerce")
    subset[target_name] = pd.to_numeric(subset[target_name], errors="coerce")
    if model_type == "sqrt_fines":
        subset["fines"] = pd.to_numeric(subset["fines"], errors="coerce")
        subset = subset.dropna(subset=["N60", target_name, "fines"])
    else:
        subset = subset.dropna(subset=["N60", target_name])

    if subset.empty:
        return None

    
    # Limpiar point_code NaN después de eliminar outliers
    if "point_code" in subset.columns:
        subset = subset.dropna(subset=["point_code"])

    analysis = analyze_target(
        subset["N60"].values,
        subset[target_name].values,
        target_name,
        model_type=model_type,
        fines=subset["fines"].values if model_type == "sqrt_fines" else None,
        fixed_c=fixed_c,
    )
    best = analysis["best"]

    ylabel = "Ángulo de fricción interna, φ (°)"
    title = "Relación entre N<sub>60</sub> y el ángulo de fricción interna, φ (°)"

    point_labels = subset["point_code"].values if "point_code" in subset.columns else None
    
    plot_kwargs = {
        "xlabel": "Número de golpes corregido, N<sub>60</sub>",
        "title": title,
        "equation_text": best.equation,
        "r2_value": best.r2,
        "rmse_value": best.rmse,
        "mape_value": best.mape,
        "point_labels": point_labels,
    }
    if model_type == "sqrt_fines":
        plot_html = plot_model_comparison_3d(
            subset["N60"].values,
            subset["fines"].values,
            subset[target_name].values,
            best,
            ylabel="Finos (FC, %)",
            zlabel=ylabel,
            **plot_kwargs,
        )
        plot_html_embed = plot_model_comparison_3d(
            subset["N60"].values,
            subset["fines"].values,
            subset[target_name].values,
            best,
            ylabel="Finos (FC, %)",
            zlabel=ylabel,
            **plot_kwargs,
            embed_mode=True,
        )
    else:
        plot_html = plot_model_comparison(
            subset["N60"].values,
            subset[target_name].values,
            analysis["results"],
            ylabel=ylabel,
            fines=subset["fines"].values if model_type == "sqrt_fines" else None,
            **plot_kwargs,
        )
        plot_html_embed = plot_model_comparison(
            subset["N60"].values,
            subset[target_name].values,
            analysis["results"],
            ylabel=ylabel,
            fines=subset["fines"].values if model_type == "sqrt_fines" else None,
            **plot_kwargs,
            embed_mode=True,
        )

    # Guardar el subconjunto usado para la regresión como Excel para Power BI
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        latest_xlsx_path = OUTPUTS_DIR / "latest_regression_data.xlsx"
        # Crear un Excel con dos hojas: 'data' (subset usado) y 'summary' (ecuación y métricas)
        summary_df = pd.DataFrame(
            {
                "metric": ["equation", "r2", "rmse", "dispersion_percent"],
                "value": [
                    best.equation,
                    float(best.r2) if hasattr(best, "r2") else None,
                    float(best.rmse) if hasattr(best, "rmse") else None,
                    float(best.mape) if hasattr(best, "mape") else None,
                ],
            }
        )
        with pd.ExcelWriter(latest_xlsx_path, engine="openpyxl") as writer:
            subset.to_excel(writer, sheet_name="data", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)
    except Exception:
        # No detener el entrenamiento si la escritura falla; simplemente ignorar
        pass

    return {
        "best": best,
        "plot_html": plot_html,
        "plot_html_embed": plot_html_embed,
        "equation": best.equation,
        "r2": best.r2,
        "rmse": best.rmse,
        "mape": best.mape,
        "model_name": best.name,
    }


def _dashboard(request, view_mode):
    sections = _base_sections()
    context = {
        "form": UploadForm(),
        "sections": list(sections.values()),
        "has_preview": False,
        "view_mode": view_mode,
        "landing_page": False,
    }

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "clear":
            request.session.pop(UPLOAD_SESSION_KEY, None)
            request.session.pop(UPLOAD_NAME_SESSION_KEY, None)
            request.session.pop(UPLOAD_SHEET_SESSION_KEY, None)
            return render(request, "dashboard.html", context)

        form = UploadForm(request.POST, request.FILES)
        context["form"] = form

        if not form.is_valid():
            context["error"] = "Sube un archivo CSV o Excel para continuar."
            return render(request, "dashboard.html", context)

        try:
            uploaded_file = form.cleaned_data.get("data_file")
            model_type = form.cleaned_data.get("model_type") or "linear"
            selected_sheet = request.POST.get("selected_sheet") or None
            if uploaded_file:
                df, sheet_names, selected_sheet = _store_uploaded_file(
                    request, uploaded_file, sheet_name=selected_sheet
                )
            else:
                df, sheet_names, selected_sheet = _load_session_dataframe(
                    request, sheet_name=selected_sheet
                )

            excluded_columns = request.POST.getlist("exclude_columns")
            df = _apply_column_exclusions(df, excluded_columns)

            excluded_columns = request.POST.getlist("exclude_columns")
            df = _apply_column_exclusions(df, excluded_columns)

            excluded_columns = request.POST.getlist("exclude_columns")
            df = _apply_column_exclusions(df, excluded_columns)

            context.update(
                _build_preview_context(
                    df,
                    request.POST,
                    sheet_names=sheet_names,
                    selected_sheet=selected_sheet,
                )
            )

            if action != "train":
                context["message"] = (
                    "Archivo cargado. Ahora selecciona qué columna corresponde a N<sub>60</sub>, φ, y opcionalmente el código de punto antes de entrenar."
                )
                return render(request, "dashboard.html", context)

            selected_n60 = request.POST.get("selected_n60", "")
            selected_phi = request.POST.get("selected_phi", "")
            selected_point_code = request.POST.get("selected_point_code", "")
            selected_fines = request.POST.get("selected_fines", "")
            fixed_c_value = None
            if model_type == "sqrt_fines":
                fixed_c_value = _parse_fixed_c(request.POST)
            if model_type == "sqrt_fines" and not selected_fines:
                selected_fines = _guess_fines_column(df.columns) or ""
            if model_type == "sqrt_fines" and not selected_fines:
                candidates = [
                    col
                    for col in df.columns
                    if col not in {selected_n60, selected_phi, selected_point_code}
                ]
                if len(candidates) == 1:
                    selected_fines = candidates[0]

            if not selected_n60:
                raise ValueError("Debes seleccionar una columna para N<sub>60</sub>.")

            if selected_n60 not in df.columns:
                raise ValueError("La columna elegida para N<sub>60</sub> no existe en el archivo.")

            for target_name, selected_target in (("phi", selected_phi),):
                if not selected_target:
                    continue

                result = _train_one_target(
                    df,
                    selected_n60,
                    selected_target,
                    target_name,
                    model_type,
                    point_code_column=selected_point_code if selected_point_code else None,
                    fines_column=selected_fines if selected_fines else None,
                    fixed_c=fixed_c_value,
                )
                if result is None:
                    continue

                best = result["best"]
                sections[target_name].update(
                    {
                        "best": best,
                        "plot_html": result["plot_html"],
                        "equation": result["equation"],
                        "r2": result["r2"],
                        "rmse": result["rmse"],
                        "mape": result["mape"],
                        "model_name": result["model_name"],
                    }
                )
                _save_latest_plot_html(
                    result["plot_html"],
                    {
                        "view_mode": "regression",
                        "equation": result["equation"],
                        "r2": result["r2"],
                        "rmse": result["rmse"],
                        "mape": result["mape"],
                        "model_name": result["model_name"],
                    },
                    plot_html_embed=result["plot_html_embed"],
                )

            context["sections"] = list(sections.values())

            if not any(item["plot_html"] for item in context["sections"]):
                context["error"] = (
                    "No se pudo entrenar ningún modelo. Verifica que N<sub>60</sub> y las columnas elegidas tengan valores numéricos."
                )
            else:
                context["message"] = "Modelo entrenado correctamente con las columnas seleccionadas."
        except Exception as exc:
            context["error"] = f"Error al procesar el archivo: {exc}"

    _hydrate_context_from_snapshot(request, context)
    return render(request, "dashboard.html", context)


def landing(request):
    return render(
        request,
        "dashboard.html",
        {
            "landing_page": True,
            "view_mode": None,
        },
    )


def regression_view(request):
    """Vista simple de regresión: upload → preview → train → resultados"""
    sections = _base_sections()
    context = {
        "form": UploadForm(),
        "sections": list(sections.values()),
        "has_preview": False,
        "view_mode": "regression",
        "landing_page": False,
    }

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "clear":
            request.session.pop(UPLOAD_SESSION_KEY, None)
            request.session.pop(UPLOAD_NAME_SESSION_KEY, None)
            request.session.pop(UPLOAD_SHEET_SESSION_KEY, None)
            return render(request, "dashboard.html", context)

        form = UploadForm(request.POST, request.FILES)
        context["form"] = form

        if not form.is_valid():
            context["error"] = "Sube un archivo CSV o Excel para continuar."
            return render(request, "dashboard.html", context)

        try:
            uploaded_file = form.cleaned_data.get("data_file")
            model_type = form.cleaned_data.get("model_type") or "linear"
            selected_sheet = request.POST.get("selected_sheet") or None
            if uploaded_file:
                df, sheet_names, selected_sheet = _store_uploaded_file(
                    request, uploaded_file, sheet_name=selected_sheet
                )
            else:
                df, sheet_names, selected_sheet = _load_session_dataframe(
                    request, sheet_name=selected_sheet
                )

            context.update(
                _build_preview_context(
                    df,
                    request.POST,
                    sheet_names=sheet_names,
                    selected_sheet=selected_sheet,
                )
            )

            if action != "train":
                context["message"] = (
                    "Archivo cargado. Ahora selecciona qué columna corresponde a N<sub>60</sub>, φ, y opcionalmente el código de punto antes de entrenar."
                )
                return render(request, "dashboard.html", context)

            selected_n60 = request.POST.get("selected_n60", "")
            selected_phi = request.POST.get("selected_phi", "")
            selected_point_code = request.POST.get("selected_point_code", "")
            selected_fines = request.POST.get("selected_fines", "")
            fixed_c_value = None
            if model_type == "sqrt_fines":
                fixed_c_value = _parse_fixed_c(request.POST)
            if model_type == "sqrt_fines" and not selected_fines:
                selected_fines = _guess_fines_column(df.columns) or ""
            if model_type == "sqrt_fines" and not selected_fines:
                candidates = [
                    col
                    for col in df.columns
                    if col not in {selected_n60, selected_phi, selected_point_code}
                ]
                if len(candidates) == 1:
                    selected_fines = candidates[0]

            if not selected_n60:
                raise ValueError("Debes seleccionar una columna para N<sub>60</sub>.")

            if selected_n60 not in df.columns:
                raise ValueError("La columna elegida para N<sub>60</sub> no existe en el archivo.")

            for target_name, selected_target in (("phi", selected_phi),):
                if not selected_target:
                    continue

                result = _train_one_target(
                    df,
                    selected_n60,
                    selected_target,
                    target_name,
                    model_type,
                    point_code_column=selected_point_code if selected_point_code else None,
                    fines_column=selected_fines if selected_fines else None,
                    fixed_c=fixed_c_value,
                )
                if result is None:
                    continue

                best = result["best"]
                sections[target_name].update(
                    {
                        "best": best,
                        "plot_html": result["plot_html"],
                        "equation": result["equation"],
                        "r2": result["r2"],
                        "rmse": result["rmse"],
                        "mape": result["mape"],
                        "model_name": result["model_name"],
                    }
                )
                _save_latest_plot_html(
                    result["plot_html"],
                    {
                        "view_mode": "regression",
                        "equation": result["equation"],
                        "r2": result["r2"],
                        "rmse": result["rmse"],
                        "mape": result["mape"],
                        "model_name": result["model_name"],
                    },
                    plot_html_embed=result["plot_html_embed"],
                )

            context["sections"] = list(sections.values())

            if not any(item["plot_html"] for item in context["sections"]):
                context["error"] = (
                    "No se pudo entrenar ningún modelo. Verifica que N<sub>60</sub> y las columnas elegidas tengan valores numéricos."
                )
            else:
                context["message"] = "Modelo entrenado correctamente con las columnas seleccionadas."
        except Exception as exc:
            context["error"] = f"Error al procesar el archivo: {exc}"

    _hydrate_context_from_snapshot(request, context)
    return render(request, "dashboard.html", context)


def comparison_view(request):
    """Vista de comparación: sube tabla con datos de autores y selecciona tu correlación"""
    target_type = request.POST.get("target_type") or request.GET.get("target_type", "phi")
    
    context = {
        "form": UploadForm(),
        "has_preview": False,
        "view_mode": "comparison",
        "landing_page": False,
        "preview_headers": [],
        "preview_rows": [],
        "comparison_plot_html": None,
        "target_type": target_type,
        "selected_fines_low": "",
        "selected_fines_high": "",
        "show_fines_band": False,
    }

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "clear":
            request.session.pop(UPLOAD_SESSION_KEY, None)
            request.session.pop(UPLOAD_NAME_SESSION_KEY, None)
            request.session.pop(UPLOAD_SHEET_SESSION_KEY, None)
            return render(request, "comparison.html", context)

        form = UploadForm(request.POST, request.FILES)
        context["form"] = form

        if not form.is_valid():
            context["error"] = "Sube un archivo CSV o Excel con datos de comparación."
            return render(request, "comparison.html", context)

        try:
            uploaded_file = form.cleaned_data.get("data_file")
            selected_sheet = request.POST.get("selected_sheet") or None
            if uploaded_file:
                df, sheet_names, selected_sheet = _store_uploaded_file(
                    request, uploaded_file, sheet_name=selected_sheet
                )
            else:
                df, sheet_names, selected_sheet = _load_session_dataframe(
                    request, sheet_name=selected_sheet
                )

            context.update(
                _build_preview_context(
                    df,
                    request.POST,
                    sheet_names=sheet_names,
                    selected_sheet=selected_sheet,
                )
            )

            if action != "generate":
                context["message"] = (
                    "Archivo cargado. Ahora selecciona cuál columna corresponde a tu correlación para generar la gráfica de comparación."
                )
                return render(request, "comparison.html", context)

            # Get selected columns
            your_column = request.POST.get("your_correlation_column", "")
            n60_col = request.POST.get("n60_column", "")
            field_spt_col = request.POST.get("field_spt_column", "")
            field_phi_col = request.POST.get("field_phi_column", "")
            fines_low_col = request.POST.get("fines_low_column", "")
            fines_high_col = request.POST.get("fines_high_column", "")
            show_fines_band = request.POST.get("show_fines_band") == "on"

            context["selected_fines_low"] = fines_low_col
            context["selected_fines_high"] = fines_high_col
            context["show_fines_band"] = show_fines_band
            
            if not your_column:
                raise ValueError("Debes seleccionar una columna para tu correlación.")
            if not n60_col:
                raise ValueError("Debes seleccionar la columna N60/SPT.")
            if not field_spt_col:
                raise ValueError("Debes seleccionar la columna de SPT de campo.")
            if not field_phi_col:
                raise ValueError("Debes seleccionar la columna de ángulo de fricción de ensayo.")

            if your_column not in df.columns:
                raise ValueError("La columna seleccionada no existe en el archivo.")
            if n60_col not in df.columns:
                raise ValueError("La columna N60/SPT no existe en el archivo.")
            if field_spt_col not in df.columns:
                raise ValueError("La columna SPT de campo no existe en el archivo.")
            if field_phi_col not in df.columns:
                raise ValueError("La columna de ángulo de fricción no existe en el archivo.")
            if show_fines_band:
                if not fines_low_col or not fines_high_col:
                    raise ValueError("Debes seleccionar las columnas de finos menor y finos mayor.")
                if fines_low_col not in df.columns or fines_high_col not in df.columns:
                    raise ValueError("Las columnas de finos seleccionadas no existen en el archivo.")

            # Convert all columns to numeric
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except:
                    pass

            # Extract numeric columns (excluding unnamed index columns)
            numeric_cols = []
            for col in df.columns:
                try:
                    if df[col].dtype in ['float64', 'int64'] and df[col].notna().sum() > 0:
                        numeric_cols.append(col)
                except:
                    pass

            excluded_columns = request.POST.getlist("exclude_columns")
            if excluded_columns:
                numeric_cols = [col for col in numeric_cols if col not in excluded_columns]

            if your_column not in numeric_cols:
                raise ValueError("La columna seleccionada no tiene datos numéricos.")
            if n60_col not in numeric_cols:
                raise ValueError("La columna N60/SPT no tiene datos numéricos.")
            if field_spt_col not in numeric_cols:
                raise ValueError("La columna SPT de campo no tiene datos numéricos.")
            if field_phi_col not in numeric_cols:
                raise ValueError("La columna de ángulo de fricción no tiene datos numéricos.")
            if show_fines_band:
                if fines_low_col not in numeric_cols or fines_high_col not in numeric_cols:
                    raise ValueError("Las columnas de finos seleccionadas no tienen datos numéricos.")

            # Build data for comparison plot
            from utils.plotting import plot_author_comparison
            
            # Create a working dataframe with only numeric columns
            df_numeric = df[[col for col in df.columns if col in numeric_cols]].copy()
            
            # Drop rows with NaN in required columns to keep alignment
            required_columns = [n60_col, your_column, field_spt_col, field_phi_col]
            if show_fines_band:
                required_columns.extend([fines_low_col, fines_high_col])
            df_numeric = df_numeric.dropna(subset=required_columns)
            
            # Sort by N60 column (ascending)
            df_numeric = df_numeric.sort_values(by=n60_col)
            
            if len(df_numeric) < 3:
                raise ValueError("No hay suficientes datos numéricos después de ordenar.")
            
            # Prepare x (N60) and y (your correlation)
            x = df_numeric[n60_col].values
            your_y = df_numeric[your_column].values

            # Prepare field points (SPT vs phi ensayo)
            field_x = df_numeric[field_spt_col].values
            field_y = df_numeric[field_phi_col].values
            
            # Prepare series data for author columns (exclude N60 and field columns)
            series_data = {}
            for col in numeric_cols:
                if col not in (n60_col, field_spt_col, field_phi_col, fines_low_col, fines_high_col):
                    # Use values from the sorted, clean dataframe
                    series_data[col] = df_numeric[col].values

            # Compute R2 for each author series vs field points
            from utils.metrics import r2_score
            r2_by_series = {}
            for col_name, col_values in series_data.items():
                if col_name == your_column:
                    continue
                r2_by_series[col_name] = r2_score(field_y, col_values)
            r2_your = r2_score(field_y, your_y)

            # Set title and ylabel based on target_type
            ylabel = "Ángulo de fricción interna, φ (°)"
            title = "Comparación de correlaciones para φ"

            # Generate comparison plot
            comparison_kwargs = {
                "x": x,
                "your_y": your_y,
                "your_column_name": your_column,
                "comparison_series": series_data,
                "field_points_x": field_x,
                "field_points_y": field_y,
                "x_label": n60_col,
                "ylabel": ylabel,
                "title": title,
                "target_type": target_type,
                "r2_by_series": r2_by_series,
                "r2_your": r2_your,
                "fines_low": df_numeric[fines_low_col].values if show_fines_band else None,
                "fines_high": df_numeric[fines_high_col].values if show_fines_band else None,
                "fines_low_label": fines_low_col or None,
                "fines_high_label": fines_high_col or None,
                "show_fines_band": show_fines_band,
            }
            plot_html = plot_author_comparison(**comparison_kwargs)
            plot_html_embed = plot_author_comparison(**comparison_kwargs, embed_mode=True)

            context["comparison_plot_html"] = plot_html
            context["message"] = "Grafica de comparacion generada correctamente."
            context["your_column"] = your_column
            context["r2_by_series"] = r2_by_series
            context["r2_your"] = r2_your
            _save_latest_plot_html(
                plot_html,
                {
                    "view_mode": "comparison",
                    "your_column": your_column,
                    "r2_by_series": r2_by_series,
                    "r2_your": r2_your,
                },
                plot_html_embed=plot_html_embed,
            )

        except Exception as exc:
            context["error"] = f"Error al procesar: {exc}"

    _hydrate_context_from_snapshot(request, context)
    return render(request, "comparison.html", context)


def fines_view(request):
    """Vista de finos: sube una tabla, elige finos y phi y genera una gráfica 2D."""
    context = {
        "form": FinesUploadForm(),
        "has_preview": False,
        "view_mode": "fines",
        "landing_page": False,
        "preview_headers": [],
        "preview_rows": [],
        "fines_plot_html": None,
        "selected_fines": "",
        "selected_phi": "",
    }

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "clear":
            request.session.pop(UPLOAD_SESSION_KEY, None)
            request.session.pop(UPLOAD_NAME_SESSION_KEY, None)
            request.session.pop(UPLOAD_SHEET_SESSION_KEY, None)
            return render(request, "fines.html", context)

        form = FinesUploadForm(request.POST, request.FILES)
        context["form"] = form

        if not form.is_valid():
            context["error"] = "Sube un archivo CSV o Excel para continuar."
            return render(request, "fines.html", context)

        try:
            uploaded_file = form.cleaned_data.get("data_file")
            selected_sheet = request.POST.get("selected_sheet") or None
            if uploaded_file:
                df, sheet_names, selected_sheet = _store_uploaded_file(
                    request, uploaded_file, sheet_name=selected_sheet
                )
            else:
                df, sheet_names, selected_sheet = _load_session_dataframe(
                    request, sheet_name=selected_sheet
                )

            excluded_columns = request.POST.getlist("exclude_columns")
            df = _apply_column_exclusions(df, excluded_columns)

            context.update(
                _build_preview_context(
                    df,
                    request.POST,
                    sheet_names=sheet_names,
                    selected_sheet=selected_sheet,
                )
            )

            if action != "generate":
                context["message"] = (
                    "Archivo cargado. Ahora selecciona la columna de finos y la columna de ángulo de fricción para generar la gráfica."
                )
                return render(request, "fines.html", context)

            selected_fines = request.POST.get("selected_fines", "")
            selected_phi = request.POST.get("selected_phi", "")

            if not selected_fines:
                selected_fines = _guess_fines_column(df.columns) or ""
            if not selected_phi:
                selected_phi = "phi" if "phi" in df.columns else ""

            if not selected_fines:
                raise ValueError("Debes seleccionar la columna de finos.")
            if not selected_phi:
                raise ValueError("Debes seleccionar la columna de ángulo de fricción.")

            if selected_fines not in df.columns:
                raise ValueError("La columna seleccionada para finos no existe en el archivo.")
            if selected_phi not in df.columns:
                raise ValueError("La columna seleccionada para φ no existe en el archivo.")

            df_numeric = df[[selected_fines, selected_phi]].copy()
            df_numeric[selected_fines] = pd.to_numeric(df_numeric[selected_fines], errors="coerce")
            df_numeric[selected_phi] = pd.to_numeric(df_numeric[selected_phi], errors="coerce")
            df_numeric = df_numeric.dropna(subset=[selected_fines, selected_phi])

            if len(df_numeric) < 2:
                raise ValueError("No hay suficientes datos numéricos para generar la gráfica.")

            fines_values = df_numeric[selected_fines].values
            phi_values = df_numeric[selected_phi].values

            fines_plot_kwargs = {
                "fines_label": "Porcentaje de finos (%)",
                "phi_label": "Ángulo de fricción interna, φ (°)",
                "title": "Relación entre porcentaje de finos y φ",
            }
            plot_html = plot_fines_phi_relationship(fines_values, phi_values, **fines_plot_kwargs)
            plot_html_embed = plot_fines_phi_relationship(
                fines_values, phi_values, **fines_plot_kwargs, embed_mode=True
            )

            context["fines_plot_html"] = plot_html
            context["message"] = "Gráfica de finos generada correctamente."
            context["selected_fines"] = selected_fines
            context["selected_phi"] = selected_phi
            _save_latest_plot_html(
                plot_html,
                {
                    "view_mode": "fines",
                    "selected_fines": selected_fines,
                    "selected_phi": selected_phi,
                },
                plot_html_embed=plot_html_embed,
            )
        except Exception as exc:
            context["error"] = f"Error al procesar: {exc}"

    _hydrate_context_from_snapshot(request, context)
    return render(request, "fines.html", context)


def _load_latest_file():
    file_bytes, filename, sheet_name = load_uploaded_file()
    if not file_bytes or not filename:
        return None, None, None
    return file_bytes, filename, sheet_name


def _load_latest_dataframe(sheet_name=None):
    file_bytes, filename, active_sheet = _load_latest_file()
    if not file_bytes or not filename:
        return None, None, None
    resolved_sheet = sheet_name or active_sheet
    df = load_dataframe_from_bytes(file_bytes, filename, sheet_name=resolved_sheet)
    return df, filename, resolved_sheet


def _load_latest_all_sheets():
    file_bytes, filename, _active_sheet = _load_latest_file()
    if not file_bytes or not filename:
        return [], filename
    return load_all_dataframes_from_bytes(file_bytes, filename), filename


def powerbi_view(request):
    """Vista pública mínima para Power BI: solo la gráfica limpia."""
    plot_html = load_plot_html_embed()
    return render(
        request,
        "powerbi.html",
        {
            "plot_html": plot_html,
            "has_plot": bool(plot_html),
        },
    )


def powerbi_excel_view(request):
    """Tabla HTML del último Excel para embeber en Power BI."""
    sheets, filename = _load_latest_all_sheets()
    preview_limit = 200
    sheet_tables = []
    for sheet_name, df in sheets:
        total_rows = len(df)
        sheet_tables.append(
            {
                "name": sheet_name,
                "total_rows": total_rows,
                "truncated": total_rows > preview_limit,
                "table_html": df.head(preview_limit).to_html(classes="data-table", index=False, border=0),
            }
        )
    return render(
        request,
        "powerbi_excel.html",
        {
            "sheet_tables": sheet_tables,
            "has_excel": bool(sheets),
            "excel_filename": filename,
            "preview_limit": preview_limit,
            "sheet_count": len(sheet_tables),
        },
    )


def powerbi_data_xlsx_view(request):
    """Excel completo con todas las hojas para Power BI."""
    file_bytes, filename, _active_sheet = _load_latest_file()
    if not file_bytes or not filename:
        return HttpResponse(
            "No hay un Excel guardado todavía.\n",
            content_type="text/plain; charset=utf-8",
            status=404,
        )
    suffix = Path(filename).suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        return HttpResponse(
            "El último archivo guardado no es Excel. Usa /powerbi/data.csv para CSV.\n",
            content_type="text/plain; charset=utf-8",
            status=400,
        )
    content_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if suffix == ".xlsx"
        else "application/vnd.ms-excel"
    )
    response = HttpResponse(file_bytes, content_type=content_type)
    response["Content-Disposition"] = f'inline; filename="{Path(filename).name}"'
    return response


def regression_data_xlsx_view(request):
    """Excel con la tabla usada en la última regresión para Power BI."""
    latest_xlsx_path = OUTPUTS_DIR / "latest_regression_data.xlsx"
    if not latest_xlsx_path.exists():
        return HttpResponse(
            "No hay datos de regresión guardados todavía.",
            content_type="text/plain; charset=utf-8",
            status=404,
        )
    with open(latest_xlsx_path, "rb") as fh:
        data = fh.read()
    response = HttpResponse(data, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = f'inline; filename="{latest_xlsx_path.name}"'
    return response


def powerbi_data_csv_view(request):
    """CSV del último Excel para importar en Power BI con 'Obtener datos desde Web'."""
    file_bytes, filename, _active_sheet = _load_latest_file()
    if not file_bytes or not filename:
        return HttpResponse(
            "No hay un Excel guardado todavía.\n",
            content_type="text/plain; charset=utf-8",
            status=404,
        )

    sheet_query = request.GET.get("sheet", "").strip()
    try:
        if sheet_query:
            resolved_sheet = resolve_sheet_query(file_bytes, filename, sheet_query)
            df = load_dataframe_from_bytes(file_bytes, filename, sheet_name=resolved_sheet)
        else:
            sheets = load_all_dataframes_from_bytes(file_bytes, filename)
            if len(sheets) == 1:
                df = sheets[0][1]
            else:
                frames = []
                for sheet_name, sheet_df in sheets:
                    tagged = sheet_df.copy()
                    tagged.insert(0, "_hoja_", sheet_name)
                    frames.append(tagged)
                df = pd.concat(frames, ignore_index=True, sort=False)
    except ValueError as exc:
        return HttpResponse(
            f"{exc}\n",
            content_type="text/plain; charset=utf-8",
            status=400,
        )

    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = 'inline; filename="ultimo_excel.csv"'
    df.to_csv(path_or_buf=response, index=False)
    return response


def index(request):
    return landing(request)
