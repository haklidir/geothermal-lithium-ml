"""
plotting.py — Manuscript figure generation.

Generates the data-dependent figures: parity plots (Fig. 10), residual
diagnostics (Fig. 11), correlation heatmap (Fig. 8), Li-vs-predictors
panel (Fig. 7), and SHAP beeswarm (Fig. 12).

Figure 9 (ML workflow schematic) is produced separately by
`scripts/make_fig9_workflow.py` using Graphviz.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from .config import HEADLINE_MODELS


def parity_4models(test_results, y_test_raw, save_path):
    """Fig. 10: Parity plots for all four models."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, m in enumerate(HEADLINE_MODELS):
        y_pred = test_results[m]["preds"]
        r = test_results[m]
        ax = axes[i]
        ax.scatter(y_test_raw, y_pred, alpha=0.85, s=70, zorder=3)
        lo = min(y_test_raw.min(), y_pred.min()) * 0.9
        hi = max(y_test_raw.max(), y_pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.7, zorder=2)
        ax.set_xlabel("True Li (mg/L)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Predicted Li (mg/L)", fontsize=12)
        ax.set_title(f"({chr(97 + i)}) {m}\nRMSE={r['rmse']:.2f}, R\u00b2={r['r2']:.3f}",
                     fontsize=11)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def residual_diagnostics(test_results, y_test_raw, save_path):
    """Fig. 11: Residual histograms and residual-vs-predicted scatter,
    for all four models in a 2-row layout."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, m in enumerate(HEADLINE_MODELS):
        y_pred = test_results[m]["preds"]
        resid = y_test_raw - y_pred

        ax_top = axes[0, i]
        ax_top.hist(resid, bins=10, alpha=0.8, edgecolor="black")
        ax_top.set_xlabel("Residual (mg/L)")
        if i == 0:
            ax_top.set_ylabel("Count")
        ax_top.set_title(f"{m}: Residual Histogram")

        ax_bot = axes[1, i]
        ax_bot.scatter(y_pred, resid, alpha=0.8, s=60)
        ax_bot.axhline(0, ls="--", c="r", alpha=0.7)
        ax_bot.set_xlabel("Predicted Li (mg/L)")
        if i == 0:
            ax_bot.set_ylabel("Residual (mg/L)")
        ax_bot.set_title(f"{m}: Residual vs Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def correlation_heatmap(X_train_raw, feature_names, save_path):
    """Fig. 8: Feature correlation heatmap on the training set."""
    corr = np.corrcoef(X_train_raw.values.T)
    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=9)
    plt.yticks(range(len(feature_names)), feature_names, fontsize=9)
    plt.title("Feature Correlation Heatmap (Training Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def li_vs_predictors(train_n, test_n, target_col, save_path):
    """Fig. 7: Li versus six key hydrogeochemical predictors,
    distinguishing training and test samples."""
    panels = [
        ("Reservoir temperature (degC)", "T (\u00b0C)", False),
        ("EC (uS/cm)", "EC (\u00b5S/cm)", True),
        ("Cl (mg/l)", "Cl (mg/L)", True),
        ("Na_plus_K", "Na+K (mg/L)", True),
        ("Boron (mg/l)", "B (mg/L)", True),
        ("SiO2 (mg/l)", "SiO\u2082 (mg/L)", True),
    ]
    # Compute Na+K on-the-fly if needed
    train_n = train_n.copy()
    test_n = test_n.copy()
    if "Na_plus_K" not in train_n.columns:
        train_n["Na_plus_K"] = train_n["Na (mg/l)"] + train_n["K (mg/l)"]
        test_n["Na_plus_K"] = test_n["Na (mg/l)"] + test_n["K (mg/l)"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    y_tr = train_n[target_col].astype(float).values
    y_te = test_n[target_col].astype(float).values

    for i, (col, xlabel, log_x) in enumerate(panels):
        ax = axes[i]
        x_tr = train_n[col].astype(float).values
        x_te = test_n[col].astype(float).values
        ax.scatter(x_tr, y_tr, alpha=0.7, label="Train", s=45)
        ax.scatter(x_te, y_te, alpha=0.85, label="Test", s=55, marker="s")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Li (mg/L)")
        ax.grid(True, alpha=0.3, linestyle=":")
    axes[0].legend(loc="upper right", framealpha=0.95)
    fig.suptitle("Li (mg/L) versus key hydrogeochemical predictors", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def shap_beeswarm(X_train_raw, y_train_raw, feature_names, save_path):
    """Fig. 12: SHAP beeswarm for XGB on the full training set."""
    import shap
    from .models import build_xgb

    xgb_full = build_xgb()
    xgb_full.fit(X_train_raw.values, np.log1p(y_train_raw))
    explainer = shap.TreeExplainer(xgb_full)
    shap_vals = explainer.shap_values(X_train_raw.values)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals, X_train_raw.values,
        feature_names=feature_names, show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def dnn_training_curves(histories, save_path):
    """Supplementary: DNN training/validation loss curves per fold."""
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for i, h in enumerate(histories):
        ax = axes[i]
        ax.plot(h["loss"], label="train", alpha=0.7)
        ax.plot(h["val_loss"], label="val", alpha=0.7)
        ax.set_title(f"Fold {i + 1} ({len(h['loss'])} ep.)")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        if i == 0:
            ax.set_ylabel("Loss")
    plt.suptitle("DNN Training Curves (Early Stopping)", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
