import re

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]", "", str(col).lower())
        if key in {"n60", "n160"}:
            mapping[col] = "N60"
        elif "phi" in key or "friccion" in key:
            mapping[col] = "phi"
    return df.rename(columns=mapping)


def ensure_n60(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    if "N60" not in df.columns:
        raise ValueError("No se encontro la columna N60.")
    df["N60"] = pd.to_numeric(df["N60"], errors="coerce")
    return df.dropna(subset=["N60"]).copy()


def prepare_target(df: pd.DataFrame, target: str):
    if target not in df.columns:
        return None
    subset = df[["N60", target]].copy()
    subset[target] = pd.to_numeric(subset[target], errors="coerce")
    subset["N60"] = pd.to_numeric(subset["N60"], errors="coerce")
    subset = subset.dropna(subset=["N60", target])
    return subset


def remove_outliers_iqr(df: pd.DataFrame, columns):
    mask = pd.Series([True] * len(df), index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = mask & df[col].between(lower, upper)
    return df.loc[mask].copy()
