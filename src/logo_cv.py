"""
logo_cv.py — Leave-One-Group-Out cross-validation.

Implements the LOGO-CV evaluation reported in Section 4.3, which holds
out one location group at a time across all 26 groups in the compiled
dataset. This evaluation was added in response to Reviewer 1.

Key result: DNN R² = 0.67 (RMSE = 2.25 mg/L), Ridge R² = −226 (linear
extrapolation collapse on held-out high-Li end-members).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .config import N_SPLITS, BATCH_SIZE, EPOCHS, PATIENCE, HEADLINE_MODELS
from .models import build_dnn, build_rf, build_xgb, build_ridge, inv_dnn
from .evaluation import scores, full_scores


def run_logo_cv(X_full, y_full, groups_full, verbose=True):
    """Run leave-one-group-out CV across all location groups.

    Parameters
    ----------
    X_full : pd.DataFrame
        Full feature matrix (train+test pooled).
    y_full : array-like
        Full target vector.
    groups_full : array-like
        Location group labels.

    Returns
    -------
    dict with keys:
        logo_preds : dict {model: out-of-fold predictions for all samples}
        per_group : pd.DataFrame
            Per-group MAE and prediction range for each model.
        summary : pd.DataFrame
            Aggregate RMSE / MAE / R² per model across all 86 OOF predictions.
    """
    logo = LeaveOneGroupOut()
    n = len(y_full)
    logo_preds = {m: np.full(n, np.nan) for m in HEADLINE_MODELS}

    unique_groups = np.unique(groups_full)
    if verbose:
        print(f"LOGO-CV over {len(unique_groups)} groups, {n} samples total")

    for fold_id, (tr_idx, te_idx) in enumerate(logo.split(X_full, y_full, groups=groups_full), 1):
        held_group = groups_full[te_idx][0]
        X_tr = X_full.iloc[tr_idx].values
        X_te = X_full.iloc[te_idx].values
        y_tr = y_full[tr_idx]

        # Standardize for DNN and Ridge
        xsc = StandardScaler()
        X_tr_s = xsc.fit_transform(X_tr)
        X_te_s = xsc.transform(X_te)
        y_tr_log = np.log1p(y_tr)
        mu = y_tr_log.mean()
        sigma = y_tr_log.std() + 1e-8

        # DNN with internal validation split for early stopping
        n_val = max(int(0.15 * len(tr_idx)), 5)
        val_pick = np.random.RandomState(42 + fold_id).choice(len(tr_idx), n_val, replace=False)
        val_mask = np.zeros(len(tr_idx), dtype=bool)
        val_mask[val_pick] = True
        y_tr_std = (y_tr_log - mu) / sigma

        dnn = build_dnn(X_tr_s.shape[1])
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                           restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=30, min_lr=1e-6, verbose=0)
        dnn.fit(X_tr_s[~val_mask], y_tr_std[~val_mask],
                validation_data=(X_tr_s[val_mask], y_tr_std[val_mask]),
                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                callbacks=[es, rlr])
        logo_preds["DNN"][te_idx] = inv_dnn(dnn.predict(X_te_s, verbose=0).ravel(), mu, sigma)

        # Tree models on raw features
        rf = build_rf()
        rf.fit(X_tr, np.log1p(y_tr))
        logo_preds["RF"][te_idx] = np.expm1(rf.predict(X_te))

        xgb = build_xgb()
        xgb.fit(X_tr, np.log1p(y_tr))
        logo_preds["XGB"][te_idx] = np.expm1(xgb.predict(X_te))

        # Ridge on standardized features
        ridge = build_ridge()
        ridge.fit(X_tr_s, np.log1p(y_tr))
        logo_preds["Ridge"][te_idx] = np.maximum(np.expm1(ridge.predict(X_te_s)), 0.0)

        if verbose:
            print(f"  fold {fold_id:>2}: held out {held_group} (n={len(te_idx)})")

    # Aggregate summary
    sum_rows = []
    for m in HEADLINE_MODELS:
        rmse, mae, r2, _ = full_scores(y_full, logo_preds[m])
        sum_rows.append([m, rmse, mae, r2])
    summary = pd.DataFrame(sum_rows, columns=["model", "RMSE", "MAE", "R2"])

    # Per-group breakdown
    per_group_rows = []
    for g in unique_groups:
        gmask = groups_full == g
        n_g = gmask.sum()
        yt = y_full[gmask]
        row = [g, n_g, yt.mean()]
        for m in HEADLINE_MODELS:
            yp = logo_preds[m][gmask]
            if np.isnan(yp).any():
                row.extend([np.nan, np.nan])
            else:
                row.extend([np.mean(np.abs(yt - yp)), np.max(np.abs(yt - yp))])
        per_group_rows.append(row)

    cols = ["group", "n", "mean_Li"]
    for m in HEADLINE_MODELS:
        cols.extend([f"{m}_MAE", f"{m}_MaxAbsErr"])
    per_group = pd.DataFrame(per_group_rows, columns=cols)

    return dict(
        logo_preds=logo_preds,
        per_group=per_group,
        summary=summary,
    )
