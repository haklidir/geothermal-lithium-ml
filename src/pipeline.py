"""
pipeline.py — End-to-end training and evaluation pipeline.

Orchestrates GroupKFold cross-validation across the four regressors (DNN,
Ridge, RF, XGB) plus ElasticNet (supplementary), aggregates per-fold
predictions via per-sample median, and sizes empirical prediction
intervals from pooled CV residuals.

This module reproduces the main computational loop of Sections 3.4-3.5
and Section 4 of the manuscript.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .config import (
    N_SPLITS, BATCH_SIZE, EPOCHS, PATIENCE, CP_COVERAGE, ALL_MODELS,
)
from .models import build_dnn, build_rf, build_xgb, build_ridge, build_elasticnet, inv_dnn
from .evaluation import scores, full_scores, cp_stats


def fit_dnn_fold(X_tr_s, y_tr, X_va_s, y_va, mu, sigma):
    """Fit DNN on one fold with early stopping; return predictions in mg/L."""
    y_tr_std = (np.log1p(y_tr) - mu) / sigma
    y_va_std = (np.log1p(y_va) - mu) / sigma
    dnn = build_dnn(X_tr_s.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                       restore_best_weights=True, verbose=0)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                            patience=30, min_lr=1e-6, verbose=0)
    hist = dnn.fit(X_tr_s, y_tr_std,
                   validation_data=(X_va_s, y_va_std),
                   epochs=EPOCHS, batch_size=BATCH_SIZE,
                   verbose=0, callbacks=[es, rlr])
    return dnn, hist.history


def run_cv_pipeline(X_train_raw, y_train_raw, groups, X_test_raw, y_test_raw,
                    n_splits=N_SPLITS, verbose=True):
    """Run the full GroupKFold pipeline across all four models + ElasticNet.

    Parameters
    ----------
    X_train_raw, X_test_raw : pd.DataFrame
        Feature matrices in mg/L scale (engineered features included).
    y_train_raw, y_test_raw : array-like
        Targets in mg/L.
    groups : array-like
        Location group labels for GroupKFold.
    n_splits : int
        Number of CV folds (default 5).
    verbose : bool
        Print per-fold progress.

    Returns
    -------
    dict with keys:
        cv_df : pd.DataFrame
            Per-fold metrics for each model.
        test_results : dict
            Per-model ensemble predictions, metrics, coverage, mean width.
        q_hat : dict
            Per-model empirical prediction interval half-width.
        histories : list
            DNN training histories per fold.
        early_stop_epochs : list
            Early stopping epoch count per fold.
        folds_list : list of (train_idx, val_idx)
    """
    splitter = GroupKFold(n_splits=n_splits)
    folds_list = list(splitter.split(X_train_raw, y_train_raw, groups=groups))

    cv_rows = []
    early_stop_epochs = []
    histories = []
    perfold_q = {m: [] for m in ALL_MODELS}
    perfold_r2 = {m: [] for m in ALL_MODELS}
    test_preds = {m: [] for m in ALL_MODELS}

    for fold_id, (tr_idx, val_idx) in enumerate(folds_list, 1):
        if verbose:
            print(f"\n{'=' * 50}\nFOLD {fold_id} "
                  f"(train={len(tr_idx)}, val={len(val_idx)})\n{'=' * 50}")

        X_tr = X_train_raw.iloc[tr_idx].values
        X_va = X_train_raw.iloc[val_idx].values
        X_te = X_test_raw.values
        y_tr = y_train_raw[tr_idx]
        y_va = y_train_raw[val_idx]

        # Shared scaler (for DNN and Ridge)
        xsc = StandardScaler()
        X_tr_s = xsc.fit_transform(X_tr)
        X_va_s = xsc.transform(X_va)
        X_te_s = xsc.transform(X_te)
        y_tr_log = np.log1p(y_tr)
        mu = y_tr_log.mean()
        sigma = y_tr_log.std() + 1e-8

        # DNN
        dnn, hist = fit_dnn_fold(X_tr_s, y_tr, X_va_s, y_va, mu, sigma)
        histories.append(hist)
        early_stop_epochs.append(len(hist["loss"]))
        y_va_dnn = inv_dnn(dnn.predict(X_va_s, verbose=0).ravel(), mu, sigma)
        y_te_dnn = inv_dnn(dnn.predict(X_te_s, verbose=0).ravel(), mu, sigma)
        if verbose:
            print(f"  DNN: {len(hist['loss'])} epochs")

        # Random Forest (on raw features)
        rf = build_rf()
        rf.fit(X_tr, np.log1p(y_tr))
        y_va_rf = np.expm1(rf.predict(X_va))
        y_te_rf = np.expm1(rf.predict(X_te))

        # XGBoost (on raw features)
        xgb = build_xgb()
        xgb.fit(X_tr, np.log1p(y_tr))
        y_va_xgb = np.expm1(xgb.predict(X_va))
        y_te_xgb = np.expm1(xgb.predict(X_te))

        # Ridge (on standardized features)
        ridge = build_ridge()
        ridge.fit(X_tr_s, np.log1p(y_tr))
        y_va_ridge = np.maximum(np.expm1(ridge.predict(X_va_s)), 0.0)
        y_te_ridge = np.maximum(np.expm1(ridge.predict(X_te_s)), 0.0)

        # ElasticNet (on standardized features)
        enet = build_elasticnet()
        enet.fit(X_tr_s, np.log1p(y_tr))
        y_va_enet = np.maximum(np.expm1(enet.predict(X_va_s)), 0.0)
        y_te_enet = np.maximum(np.expm1(enet.predict(X_te_s)), 0.0)

        preds_map = {
            "DNN": (y_va_dnn, y_te_dnn),
            "RF": (y_va_rf, y_te_rf),
            "XGB": (y_va_xgb, y_te_xgb),
            "Ridge": (y_va_ridge, y_te_ridge),
            "ElasticNet": (y_va_enet, y_te_enet),
        }
        for m, (yv, yt) in preds_map.items():
            rmse, mae, r2 = scores(y_va, yv)
            cv_rows.append([m, fold_id, rmse, mae, r2, len(val_idx)])
            perfold_r2[m].append(r2)
            resid = np.abs(y_va - yv)
            perfold_q[m].append(np.quantile(resid, CP_COVERAGE))
            test_preds[m].append(yt)

    cv_df = pd.DataFrame(
        cv_rows, columns=["model", "fold", "rmse", "mae", "r2", "n_val"]
    )

    # Empirical PI thresholds: median over fold-wise quantiles
    q_hat = {m: float(np.median(perfold_q[m])) for m in ALL_MODELS}

    # Ensemble test predictions (per-sample median over folds)
    test_results = {}
    for m in ALL_MODELS:
        y_pred = np.median(np.vstack(test_preds[m]), axis=0)
        rmse, mae, r2, mape = full_scores(y_test_raw, y_pred)
        _, _, cov, width = cp_stats(y_test_raw, y_pred, q_hat[m])
        test_results[m] = dict(
            preds=y_pred, rmse=rmse, mae=mae, r2=r2, mape=mape,
            cov=cov, width=width,
        )

    return dict(
        cv_df=cv_df,
        test_results=test_results,
        q_hat=q_hat,
        histories=histories,
        early_stop_epochs=early_stop_epochs,
        folds_list=folds_list,
    )
