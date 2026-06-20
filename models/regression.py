from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import stats

from utils.metrics import mape, mse, r2_score, rmse


@dataclass
class ModelResult:
    name: str
    model_type: str
    equation: str
    r2: float
    mse: float
    rmse: float
    mape: float
    params: dict
    prediction_interval: Optional[np.ndarray] = field(default=None, repr=False)
    cooks_distance: Optional[np.ndarray] = field(default=None, repr=False)


def _calculate_prediction_interval(
    x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, model_dof: int, x_line: np.ndarray
) -> np.ndarray:
    """Calcula el intervalo de predicción para un modelo de regresión."""
    n = len(y)
    if n <= model_dof:
        return None

    # Suma de los cuadrados de los residuos
    sse = np.sum((y - y_pred) ** 2)
    # Error estándar de los residuos
    s_err = np.sqrt(sse / (n - model_dof))

    # Valor crítico de t para un intervalo de confianza del 95%
    t_crit = stats.t.ppf(0.975, df=n - model_dof)

    # Para la predicción en nuevos puntos x_line
    x_mean = np.mean(x)
    ssx = np.sum((x - x_mean) ** 2)

    # Distancia de los nuevos puntos a la media
    dist = (x_line - x_mean) ** 2 / ssx

    # Margen de error
    margin = t_crit * s_err * np.sqrt(1 + 1 / n + dist)
    return margin




def _calculate_cooks_distance(y, y_pred, design_matrix):
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    design_matrix = np.asarray(design_matrix, dtype=float)
    n = len(y)
    p = design_matrix.shape[1]
    if n <= p:
        return np.zeros(n)

    residuals = y - y_pred
    sse = np.sum(residuals**2)
    if sse == 0.0:
        return np.zeros(n)

    mse_val = sse / float(n - p)
    xtx_inv = np.linalg.pinv(design_matrix.T @ design_matrix)
    hat_matrix_diag = np.sum(design_matrix @ xtx_inv * design_matrix, axis=1)
    denom = (1.0 - hat_matrix_diag) ** 2
    denom[denom == 0.0] = np.inf
    cooks = (residuals**2 / (p * mse_val)) * (hat_matrix_diag / denom)
    return cooks


def _build_result(name: str, model_type: str, equation: str, y, y_pred, params, cooks_distance=None):
    return ModelResult(
        name=name,
        model_type=model_type,
        equation=equation,
        r2=r2_score(y, y_pred),
        mse=mse(y, y_pred),
        rmse=rmse(y, y_pred),
        mape=mape(y, y_pred),
        params=params,
        cooks_distance=cooks_distance,
    )


def _format_signed_term(value: float, term: str) -> str:
    sign = "-" if value < 0 else "+"
    return f"{sign} {abs(value):.4f}{term}"


def fit_linear(x, y) -> ModelResult:
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    equation = f"y = {a:.4f}*x + {b:.4f}"
    design = np.column_stack([x, np.ones_like(x)])
    result = _build_result(
        "Lineal",
        "linear",
        equation,
        y,
        y_pred,
        {"a": a, "b": b},
        cooks_distance=_calculate_cooks_distance(y, y_pred, design),
    )
    
    x_line = np.linspace(np.min(x), np.max(x), 200)
    pred_interval = _calculate_prediction_interval(x, y, y_pred, 2, x_line)
    if pred_interval is not None:
        y_line = a * x_line + b
        result.prediction_interval = np.array([y_line - pred_interval, y_line + pred_interval])
        
    return result

def fit_quadratic(x, y) -> ModelResult:
    a, b, c = np.polyfit(x, y, 2)
    y_pred = a * x**2 + b * x + c
    equation = f"y = {a:.4f}*x^2 + {b:.4f}*x + {c:.4f}"
    X_design = np.vstack([x**2, x, np.ones(len(x))]).T
    y_pred_recalc = X_design @ np.array([a, b, c])
    result = _build_result(
        "Polinómica grado 2",
        "quadratic",
        equation,
        y,
        y_pred,
        {"a": a, "b": b, "c": c},
        cooks_distance=_calculate_cooks_distance(y, y_pred_recalc, X_design),
    )

    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = np.polyval([a, b, c], x_line)

    pred_interval = _calculate_prediction_interval(x, y, y_pred_recalc, 3, x_line)
    if pred_interval is not None:
        result.prediction_interval = np.array([y_line - pred_interval, y_line + pred_interval])
        
    return result

def fit_sqrt(x, y) -> ModelResult:
    if np.any(x < 0):
        raise ValueError("No se permite N60 negativo para regresion raiz cuadrada.")
    x_sqrt = np.sqrt(x)
    a, b = np.polyfit(x_sqrt, y, 1)
    y_pred = a * x_sqrt + b
    equation = f"y = {a:.4f}*sqrt(x) + {b:.4f}"
    design = np.column_stack([x_sqrt, np.ones_like(x_sqrt)])
    result = _build_result(
        "Raiz cuadrada",
        "sqrt",
        equation,
        y,
        y_pred,
        {"a": a, "b": b},
        cooks_distance=_calculate_cooks_distance(y, y_pred, design),
    )

    x_line = np.linspace(np.min(x), np.max(x), 200)
    x_line_sqrt = np.sqrt(x_line)
    y_line = a * x_line_sqrt + b
    
    pred_interval = _calculate_prediction_interval(x_sqrt, y, y_pred, 2, x_line_sqrt)
    if pred_interval is not None:
        result.prediction_interval = np.array([y_line - pred_interval, y_line + pred_interval])
        
    return result


def fit_sqrt_fines(x, fines, y, fixed_c: float | None = None) -> ModelResult:
    x = np.asarray(x, dtype=float)
    fines = np.asarray(fines, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x < 0):
        raise ValueError("No se permite N60 negativo para regresion raiz cuadrada.")
    x_sqrt = np.sqrt(x)
    if fixed_c is None:
        design = np.column_stack([x_sqrt, fines, np.ones_like(x_sqrt)])
        coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        a, b, c = coeffs
        y_pred = design @ coeffs
    else:
        design = np.column_stack([x_sqrt, fines])
        coeffs, _, _, _ = np.linalg.lstsq(design, y - fixed_c, rcond=None)
        a, b = coeffs
        c = float(fixed_c)
        y_pred = design @ coeffs + c
    equation = f"y = {a:.4f}*sqrt(x) + {b:.4f}*fines + {c:.4f}"
    return _build_result(
        "Raiz cuadrada + finos",
        "sqrt_fines",
        equation,
        y,
        y_pred,
        {"a": a, "b": b, "c": c, "fixed_c": fixed_c},
        cooks_distance=_calculate_cooks_distance(y, y_pred, design),
    )


def fit_sqrt_log_fines(x, fines, y) -> ModelResult:
    x = np.asarray(x, dtype=float)
    fines = np.asarray(fines, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x < 0):
        raise ValueError("No se permite N60 negativo para regresion raiz cuadrada.")
    if np.any(fines <= 0):
        raise ValueError("Finos debe ser mayor que cero para regresion logaritmica.")
    x_sqrt = np.sqrt(x)
    fines_log = np.log(fines)
    design = np.column_stack([x_sqrt, fines_log, np.ones_like(x_sqrt)])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    a, b, c = coeffs
    y_pred = design @ coeffs
    equation = f"y = {a:.4f}*sqrt(x) + {b:.4f}*ln(fines) + {c:.4f}"
    return _build_result(
        "Raiz cuadrada + log(finos)",
        "sqrt_log_fines",
        equation,
        y,
        y_pred,
        {"a": a, "b": b, "c": c},
        cooks_distance=_calculate_cooks_distance(y, y_pred, design),
    )


def predict(result: ModelResult, x, fines=None):
    if result.model_type == "linear":
        a = result.params["a"]
        b = result.params["b"]
        return a * x + b
    if result.model_type == "quadratic":
        a = result.params["a"]
        b = result.params["b"]
        c = result.params["c"]
        return a * x**2 + b * x + c
    if result.model_type == "sqrt":
        a = result.params["a"]
        b = result.params["b"]
        return a * np.sqrt(x) + b
    if result.model_type == "sqrt_fines":
        if fines is None:
            raise ValueError("Se requiere fines para predecir con este modelo.")
        params = result.params
        return params["a"] * np.sqrt(x) + params["b"] * fines + params["c"]
    if result.model_type == "sqrt_log_fines":
        if fines is None:
            raise ValueError("Se requiere fines para predecir con este modelo.")
        params = result.params
        return params["a"] * np.sqrt(x) + params["b"] * np.log(fines) + params["c"]
    raise ValueError("Modelo no soportado")


def fit_models(
    x: np.ndarray,
    y: np.ndarray,
    model_type: str = "linear",
    fines: np.ndarray | None = None,
    fixed_c: float | None = None,
) -> List[ModelResult]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    results: List[ModelResult] = []

    if model_type == "quadratic":
        results.append(fit_quadratic(x, y))
    elif model_type == "sqrt":
        results.append(fit_sqrt(x, y))
    elif model_type == "sqrt_fines":
        if fines is None:
            raise ValueError("Se requiere la columna de finos para este modelo.")
        results.append(fit_sqrt_fines(x, fines, y, fixed_c=fixed_c))
    elif model_type == "sqrt_log_fines":
        if fines is None:
            raise ValueError("Se requiere la columna de finos para este modelo.")
        results.append(fit_sqrt_log_fines(x, fines, y))
    else:
        results.append(fit_linear(x, y))

    return results


def analyze_target(
    x: np.ndarray,
    y: np.ndarray,
    target: str,
    model_type: str = "linear",
    fines: np.ndarray | None = None,
    fixed_c: float | None = None,
):
    results = fit_models(x, y, model_type=model_type, fines=fines, fixed_c=fixed_c)
    target_label = "φ" if target.lower() == "phi" else target
    for result in results:
        if result.model_type == "linear":
            a = result.params["a"]
            b = result.params["b"]
            result.equation = (
                f"{target_label} = {a:.4f}*N<sub>60</sub> {_format_signed_term(b, '')}"
            )
        elif result.model_type == "quadratic":
            a = result.params["a"]
            b = result.params["b"]
            c = result.params["c"]
            result.equation = (
                f"{target_label} = {a:.4f}*N<sub>60</sub><sup>2</sup> "
                f"{_format_signed_term(b, '*N<sub>60</sub>')} "
                f"{_format_signed_term(c, '')}"
            )
        elif result.model_type == "sqrt":
            a = result.params["a"]
            b = result.params["b"]
            result.equation = (
                f"{target_label} = {a:.4f}*sqrt(N<sub>60</sub>) {_format_signed_term(b, '')}"
            )
        elif result.model_type == "sqrt_fines":
            a = result.params["a"]
            b = result.params["b"]
            c = result.params["c"]
            result.equation = (
                f"{target_label} = {a:.4f}*sqrt(N<sub>60</sub>) "
                f"{_format_signed_term(b, '*FC')} "
                f"{_format_signed_term(c, '')}"
            )
        elif result.model_type == "sqrt_log_fines":
            a = result.params["a"]
            b = result.params["b"]
            c = result.params["c"]
            result.equation = (
                f"{target_label} = {a:.4f}*sqrt(N<sub>60</sub>) "
                f"{_format_signed_term(b, '*ln(FC)')} "
                f"{_format_signed_term(c, '')}"
            )
    best = max(results, key=lambda item: item.r2)
    return {"target": target, "results": results, "best": best}
