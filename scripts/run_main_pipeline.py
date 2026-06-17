"""
run_main_pipeline.py — Reproduce the main results of the paper.

Loads the training and test CSVs, runs five-fold GroupKFold CV across the
four models, computes empirical prediction intervals, generates Tables
3–5 and all data-dependent figures, and saves everything to `results/`.

Usage
-----
    python scripts/run_main_pipeline.py \\
        --train data/training_dataset_lit.csv \\
        --test  data/testing_dataset_lit.csv \\
        --out   results

If `--train` and `--test` are omitted, the script looks under `data/`.
"""

import argparse
import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_datasets
from src.pipeline import run_cv_pipeline
from src.models import set_seeds
from src.plotting import (
    parity_4models, residual_diagnostics, correlation_heatmap,
    li_vs_predictors, shap_beeswarm, dnn_training_curves,
)

import pandas as pd


def main(args):
    set_seeds()
    out_tables = pathlib.Path(args.out) / "tables"
    out_figures = pathlib.Path(args.out) / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from\n  train: {args.train}\n  test:  {args.test}")
    data = load_datasets(args.train, args.test)
    print(f"Train: {data['X_train_raw'].shape}, Test: {data['X_test_raw'].shape}")
    print(f"Features ({len(data['feature_names_ext'])}): {data['feature_names_ext']}")

    # Main CV + test evaluation
    out = run_cv_pipeline(
        data["X_train_raw"], data["y_train_raw"], data["groups"],
        data["X_test_raw"], data["y_test_raw"],
    )

    cv_df = out["cv_df"]
    test_results = out["test_results"]
    q_hat = out["q_hat"]

    # Table 3 — test metrics
    t3_rows = []
    for m in ["DNN", "Ridge", "RF", "XGB"]:
        r = test_results[m]
        t3_rows.append([
            m,
            f"{r['rmse']:.3f}", f"{r['mae']:.3f}", f"{r['r2']:.3f}",
            f"{r['mape']:.1f}", f"{r['cov'] * 100:.1f}%", f"{r['width']:.3f}",
        ])
    t3 = pd.DataFrame(t3_rows, columns=[
        "Model", "RMSE (mg/L)", "MAE (mg/L)", "R\u00b2", "MAPE (%)",
        "Empirical Coverage", "Mean Width (mg/L)",
    ])
    t3.to_csv(out_tables / "TABLE3_test_metrics.csv", index=False)
    print("\nTABLE 3 — Test-set metrics:")
    print(t3.to_string(index=False))

    # Table 4 — per-fold CV
    cv_df.to_csv(out_tables / "TABLE4_cv_perfold.csv", index=False)

    # Table 5 — CV summary mean ± std
    t5_rows = []
    for m in ["DNN", "Ridge", "RF", "XGB"]:
        sub = cv_df[cv_df["model"] == m]
        t5_rows.append([
            m,
            f"{sub['rmse'].mean():.3f} \u00b1 {sub['rmse'].std():.3f}",
            f"{sub['mae'].mean():.3f} \u00b1 {sub['mae'].std():.3f}",
            f"{sub['r2'].mean():.3f} \u00b1 {sub['r2'].std():.3f}",
        ])
    t5 = pd.DataFrame(t5_rows, columns=[
        "Model", "RMSE (mean\u00b1std)", "MAE (mean\u00b1std)", "R\u00b2 (mean\u00b1std)"
    ])
    t5.to_csv(out_tables / "TABLE5_cv_summary.csv", index=False)
    print("\nTABLE 5 — CV summary:")
    print(t5.to_string(index=False))

    # Save all predictions
    pred_data = {"y_true": data["y_test_raw"]}
    for m in ["DNN", "Ridge", "RF", "XGB"]:
        pred_data[f"pred_{m}"] = test_results[m]["preds"]
        pred_data[f"pi_lo_{m}"] = test_results[m]["preds"] - q_hat[m]
        pred_data[f"pi_hi_{m}"] = test_results[m]["preds"] + q_hat[m]
    pd.DataFrame(pred_data).to_csv(out_tables / "predictions_all.csv", index=False)

    # Figures
    print("\nGenerating figures...")
    li_vs_predictors(
        data["train_n"], data["test_n"], data["target"],
        out_figures / "Fig7_Li_vs_predictors.png",
    )
    correlation_heatmap(
        data["X_train_raw"], data["feature_names_ext"],
        out_figures / "Fig8_corr_heatmap.png",
    )
    parity_4models(
        test_results, data["y_test_raw"],
        out_figures / "Fig10_parity_4models.png",
    )
    residual_diagnostics(
        test_results, data["y_test_raw"],
        out_figures / "Fig11_residuals_4models.png",
    )
    shap_beeswarm(
        data["X_train_raw"], data["y_train_raw"], data["feature_names_ext"],
        out_figures / "Fig12_shap_beeswarm.png",
    )
    dnn_training_curves(out["histories"], out_figures / "FigA_dnn_training_curves.png")

    print(f"\nDone. Results saved under: {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/training_dataset_lit.csv",
                        help="Path to training CSV")
    parser.add_argument("--test", default="data/testing_dataset_lit.csv",
                        help="Path to test CSV")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()
    main(args)
