import argparse
from pathlib import Path

import pandas as pd

from models.regression import analyze_target
from utils.io import load_dataframe
from utils.plotting import fig_to_base64, plot_model_comparison, save_fig
from utils.validation import ensure_n60, prepare_target, remove_outliers_iqr


def run_analysis(file_path: str, outdir: str) -> int:
    df = load_dataframe(file_path)
    df = ensure_n60(df)

    results = []
    for target in ("phi",):
        subset = prepare_target(df, target)
        if subset is None or subset.empty:
            continue
        subset = remove_outliers_iqr(subset, ["N60", target])
        analysis = analyze_target(subset["N60"].values, subset[target].values, target)
        results.append(analysis)

        fig = plot_model_comparison(
            subset["N60"].values,
            subset[target].values,
            analysis["results"],
            xlabel="N60",
            ylabel=target,
        )
        out_path = Path(outdir) / f"{target.lower()}_modelos.png"
        save_fig(fig, out_path)

    if not results:
        print("No hay columna phi para entrenar modelos.")
        return 1

    for analysis in results:
        best = analysis["best"]
        print(f"\nObjetivo: {analysis['target']}")
        print(f"Mejor modelo: {best.name}")
        print(f"Ecuacion: {best.equation}")
        print(f"R2: {best.r2:.4f}  RMSE: {best.rmse:.4f}")
        print("Modelos:")
        for res in analysis["results"]:
            print(
                f"- {res.name}: {res.equation} | R2={res.r2:.4f} | RMSE={res.rmse:.4f}"
            )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analisis de regresion N60")
    parser.add_argument("--file", required=True, help="Ruta a CSV o Excel")
    parser.add_argument("--outdir", default="outputs", help="Carpeta de salida")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    return run_analysis(args.file, args.outdir)


if __name__ == "__main__":
    raise SystemExit(main())
