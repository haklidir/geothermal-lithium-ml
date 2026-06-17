"""
run_logo_cv.py — Reproduce the leave-one-group-out CV results (Section 4.3).

Pools training and test samples (n=86), runs LOGO-CV across all 26 location
groups, and reports per-model aggregate metrics plus a per-group breakdown
showing where each model succeeds or fails.

Headline result: DNN R² = 0.67 (RMSE = 2.25 mg/L), Ridge R² = −226
(linear extrapolation collapse, dominated by the Tuzla T1 fold).
"""

import argparse
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.preprocessing import load_datasets, build_features
from src.logo_cv import run_logo_cv
from src.models import set_seeds


def main(args):
    set_seeds()
    out_tables = pathlib.Path(args.out) / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    data = load_datasets(args.train, args.test)

    # Pool train + test
    X_full = pd.concat(
        [data["X_train_raw"], data["X_test_raw"]], axis=0, ignore_index=True
    )
    y_full = np.concatenate([data["y_train_raw"], data["y_test_raw"]])
    groups_full = np.concatenate([
        data["train_n"][data["group_col"]].astype(str).values,
        data["test_n"][data["group_col"]].astype(str).values,
    ])

    print(f"Pooled dataset: {len(y_full)} samples, "
          f"{len(np.unique(groups_full))} location groups")

    out = run_logo_cv(X_full, y_full, groups_full)

    print("\n" + "=" * 70)
    print("LOGO-CV SUMMARY")
    print("=" * 70)
    print(out["summary"].to_string(index=False, float_format="%.3f"))
    out["summary"].to_csv(out_tables / "logo_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("PER-GROUP MAE")
    print("=" * 70)
    print(out["per_group"].sort_values("DNN_MAE", ascending=False)
          .to_string(index=False, float_format="%.3f"))
    out["per_group"].to_csv(out_tables / "logo_per_group.csv", index=False)

    # Save all OOF predictions
    pred_df = pd.DataFrame({"group": groups_full, "y_true": y_full})
    for m in ["DNN", "Ridge", "RF", "XGB"]:
        pred_df[f"pred_{m}"] = out["logo_preds"][m]
    pred_df.to_csv(out_tables / "logo_predictions.csv", index=False)

    print(f"\nDone. Outputs under: {out_tables}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/training_dataset_lit.csv")
    parser.add_argument("--test", default="data/testing_dataset_lit.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    main(args)
