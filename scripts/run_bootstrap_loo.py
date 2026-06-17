"""
run_bootstrap_loo.py — Reproduce the DNN-vs-Ridge bootstrap test and
the leave-one-out test-set sensitivity analysis (Section 4.2).

Generates:
- DNN vs Ridge MAE difference with paired bootstrap 95% CI and p-value.
- LOO sensitivity table showing how dropping each test sample affects
  DNN and Ridge R² (used to flag the Tuzla T1 dominance).
"""

import argparse
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.preprocessing import load_datasets
from src.pipeline import run_cv_pipeline
from src.evaluation import paired_bootstrap
from src.models import set_seeds


def main(args):
    set_seeds()
    out_tables = pathlib.Path(args.out) / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    data = load_datasets(args.train, args.test)
    out = run_cv_pipeline(
        data["X_train_raw"], data["y_train_raw"], data["groups"],
        data["X_test_raw"], data["y_test_raw"], verbose=False,
    )
    test_results = out["test_results"]
    y_te = data["y_test_raw"]

    y_pred_dnn = test_results["DNN"]["preds"]
    y_pred_ridge = test_results["Ridge"]["preds"]
    abs_err_dnn = np.abs(y_te - y_pred_dnn)
    abs_err_ridge = np.abs(y_te - y_pred_ridge)

    # Paired bootstrap (DNN vs Ridge)
    boot = paired_bootstrap(abs_err_dnn, abs_err_ridge, n_iter=10000)

    print("=" * 60)
    print("PAIRED BOOTSTRAP: DNN vs Ridge")
    print("=" * 60)
    print(f"Observed MAE diff (Ridge - DNN): {boot['observed_diff']:.4f} mg/L")
    print(f"95% CI: [{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]")
    print(f"p-value: {boot['p_value']:.4f}")
    sig = ("SIGNIFICANT" if (boot["p_value"] < 0.05 and boot["ci_lo"] > 0)
           else "NOT significant")
    print(f"=> DNN advantage is {sig} at alpha=0.05")

    pd.DataFrame([boot]).to_csv(out_tables / "bootstrap_dnn_vs_ridge.csv", index=False)

    # LOO sensitivity
    print("\n" + "=" * 60)
    print("LEAVE-ONE-OUT SENSITIVITY")
    print("=" * 60)
    sid_col = "Sample ID" if "Sample ID" in data["test_n"].columns else None
    loo_rows = []
    for i in range(len(y_te)):
        mask = np.ones(len(y_te), dtype=bool)
        mask[i] = False
        r2d = r2_score(y_te[mask], y_pred_dnn[mask])
        r2r = r2_score(y_te[mask], y_pred_ridge[mask])
        sid = data["test_n"].iloc[i][sid_col] if sid_col else f"s{i}"
        loo_rows.append([sid, y_te[i], r2d, r2r, r2d - r2r])
    loo_df = pd.DataFrame(
        loo_rows,
        columns=["Sample", "Li_true", "DNN_R2", "Ridge_R2", "Diff"],
    )
    print(loo_df.to_string(index=False, float_format="%.4f"))
    loo_df.to_csv(out_tables / "loo_sensitivity.csv", index=False)

    tuzla_i = int(np.argmax(y_te))
    print(f"\nHighest-Li: {loo_df.iloc[tuzla_i]['Sample']} "
          f"(Li={y_te[tuzla_i]:.1f})")
    print(f"  Without it: DNN R\u00b2={loo_df.iloc[tuzla_i]['DNN_R2']:.4f}, "
          f"Ridge R\u00b2={loo_df.iloc[tuzla_i]['Ridge_R2']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/training_dataset_lit.csv")
    parser.add_argument("--test", default="data/testing_dataset_lit.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    main(args)
