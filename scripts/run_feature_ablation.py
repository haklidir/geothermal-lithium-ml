"""
run_feature_ablation.py — Feature engineering ablation (Section 3.2).

Compares three feature configurations:
    - raw_only (8 base features)
    - engineered_only (8 features: T, pH, EC, ratios, B, SiO2)
    - combined (11 features: raw + ratios)

Reports CV mean R² and test-set R² for RF, XGB, and DNN under each
configuration. Result: combined set yields best/near-best performance
for all models (e.g. RF R² 0.57 → 0.62 when ratios are added).
"""

import argparse
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

from src.preprocessing import load_datasets, build_features
from src.models import build_dnn, build_rf, build_xgb, inv_dnn, set_seeds
from src.evaluation import full_scores
from src.config import N_SPLITS, BATCH_SIZE, EPOCHS, PATIENCE


def evaluate_config(label, fmode, train_n, test_n, feat_names, mg_col,
                    y_train_raw, y_test_raw, groups, folds_list):
    rows = []
    Xtr = build_features(train_n, feat_names, mg_col, feature_mode=fmode)
    Xte = build_features(test_n, feat_names, mg_col, feature_mode=fmode)
    print(f"\n{label}: {list(Xtr.columns)}")

    for mname in ["RF", "XGB", "DNN"]:
        fold_preds = []
        fold_r2s = []
        for tr_i, va_i in folds_list:
            if mname == "DNN":
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(Xtr.iloc[tr_i].values)
                Xva_s = sc.transform(Xtr.iloc[va_i].values)
                Xte_s = sc.transform(Xte.values)
                yl = np.log1p(y_train_raw[tr_i])
                mu, si = yl.mean(), yl.std() + 1e-8
                d = build_dnn(Xtr_s.shape[1])
                d.fit(Xtr_s, (yl - mu) / si,
                      validation_data=(Xva_s, (np.log1p(y_train_raw[va_i]) - mu) / si),
                      epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                              restore_best_weights=True, verbose=0)])
                fold_preds.append(inv_dnn(d.predict(Xte_s, verbose=0).ravel(), mu, si))
                fold_r2s.append(r2_score(y_train_raw[va_i],
                                         inv_dnn(d.predict(Xva_s, verbose=0).ravel(), mu, si)))
            else:
                M = build_rf() if mname == "RF" else build_xgb()
                M.fit(Xtr.iloc[tr_i].values, np.log1p(y_train_raw[tr_i]))
                fold_preds.append(np.expm1(M.predict(Xte.values)))
                fold_r2s.append(r2_score(y_train_raw[va_i],
                                         np.expm1(M.predict(Xtr.iloc[va_i].values))))

        y_ens = np.median(np.vstack(fold_preds), axis=0)
        rmse, mae, r2, _ = full_scores(y_test_raw, y_ens)
        rows.append([label, mname, Xtr.shape[1], rmse, mae, r2, np.mean(fold_r2s)])
    return rows


def main(args):
    set_seeds()
    out_tables = pathlib.Path(args.out) / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    data = load_datasets(args.train, args.test)
    splitter = GroupKFold(n_splits=N_SPLITS)
    folds_list = list(splitter.split(
        data["X_train_raw"], data["y_train_raw"], groups=data["groups"]
    ))

    all_rows = []
    for label, fmode in [
        ("raw_only (8)", "raw_only"),
        ("engineered_only (8)", "engineered_only"),
        ("combined (11)", "combined"),
    ]:
        all_rows.extend(evaluate_config(
            label, fmode, data["train_n"], data["test_n"],
            data["feat_names"], data["mg_col"],
            data["y_train_raw"], data["y_test_raw"], data["groups"], folds_list,
        ))

    df = pd.DataFrame(all_rows, columns=[
        "Config", "Model", "n_feat", "RMSE", "MAE", "R2", "CV_R2_mean",
    ])
    print("\n" + "=" * 60 + "\nFEATURE ABLATION RESULTS\n" + "=" * 60)
    print(df.round(4).to_string(index=False))
    df.to_csv(out_tables / "feature_ablation.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/training_dataset_lit.csv")
    parser.add_argument("--test", default="data/testing_dataset_lit.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    main(args)
