import io
from pathlib import Path

import pandas as pd

from utils.validation import normalize_columns


def load_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Formato no soportado. Usa CSV o Excel.")

    return normalize_columns(df)


def _make_unique_headers(headers):
    seen = {}
    result = []
    for header in headers:
        base = header or "Columna"
        count = seen.get(base, 0)
        if count:
            result.append(f"{base} {count + 1}")
        else:
            result.append(base)
        seen[base] = count + 1
    return result


def _maybe_use_first_row_as_header(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    columns = [str(col).strip() for col in df.columns]
    unnamed_count = sum(col.lower().startswith("unnamed") for col in columns)
    if unnamed_count < max(1, len(columns) // 2):
        return df

    first_row = df.iloc[0].tolist()
    candidates = [str(value).strip() for value in first_row]
    usable = [value for value in candidates if value and value.lower() != "nan"]
    if len(usable) < max(2, len(candidates) // 3):
        return df

    new_headers = _make_unique_headers(candidates)
    df = df.iloc[1:].copy()
    df.columns = new_headers
    return df


def get_excel_sheet_names_from_bytes(file_bytes: bytes, filename: str) -> list[str]:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        return []
    buffer = io.BytesIO(file_bytes)
    excel = pd.ExcelFile(buffer)
    return list(excel.sheet_names)


def load_dataframe_from_bytes(
    file_bytes: bytes,
    filename: str,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    if suffix == ".csv":
        df = pd.read_csv(buffer, header=0)
    elif suffix in {".xlsx", ".xls"}:
        excel = pd.ExcelFile(buffer)
        resolved_sheet = sheet_name
        if resolved_sheet is None:
            resolved_sheet = excel.sheet_names[0] if excel.sheet_names else None
        if resolved_sheet is None:
            raise ValueError("El archivo Excel no tiene hojas disponibles.")
        if resolved_sheet not in excel.sheet_names:
            raise ValueError("La hoja seleccionada no existe en el archivo.")
        df = excel.parse(sheet_name=resolved_sheet, header=0)
    else:
        raise ValueError("Formato no soportado. Usa CSV o Excel.")

    # IMPORTANTE: No normalizar nombres de columnas para preservar encabezados originales.
    # Si el archivo trae columnas "Unnamed" pero la primera fila tiene texto, usa esa fila
    # como encabezado para mostrar nombres reales.
    return _maybe_use_first_row_as_header(df)
