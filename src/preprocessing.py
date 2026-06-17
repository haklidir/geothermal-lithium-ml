"""
preprocessing.py — Data loading and feature engineering.

Implements the harmonization pipeline described in Section 3.2 of the
manuscript, including the 8 base hydrogeochemical features and 3
engineered ratio features (Na/K, Na+K, Cl/EC).
"""

import re
import numpy as np
import pandas as pd


def normalize_colnames(df):
    """Normalize Unicode in column headers (µ → u, ° → deg, whitespace collapse)."""
    cols = []
    for c in df.columns:
        s = c.replace("\u00b5", "u").replace("\u03bc", "u").replace("\u00b0", "deg")
        s = re.sub(r"\s+", " ", s).strip()
        cols.append(s)
    out = df.copy()
    out.columns = cols
    return out


def pick(df, wanted, alts=None):
    """Robust column lookup with case- and punctuation-insensitive fallback."""
    alts = alts or []
    if wanted in df.columns:
        return wanted
    for a in alts:
        if a in df.columns:
            return a

    def canon(x):
        return re.sub(r"[^a-z0-9]", "", x.lower())

    w = canon(wanted)
    for c in df.columns:
        if canon(c) == w:
            return c
    raise KeyError(f"Column not found: {wanted}")


def resolve_feature_names(df):
    """Resolve the 8 base feature column names and the target/group columns
    for a normalized dataframe.

    Returns
    -------
    dict with keys: feat_names (list[str]), target (str), mg_col (str),
    group_col (str)
    """
    feat_names = [
        pick(df, "Reservoir temperature (degC)", ["Reservoir temperature (\u00b0C)"]),
        pick(df, "pH"),
        pick(df, "EC (uS/cm)", ["EC (microS/cm)"]),
        pick(df, "K (mg/l)"),
        pick(df, "Na (mg/l)"),
        pick(df, "Boron (mg/l)"),
        pick(df, "SiO2 (mg/l)"),
        pick(df, "Cl (mg/l)"),
    ]
    target = pick(df, "Li (mg/l)")
    mg_col = pick(df, "Mg (mg/l)")
    return dict(feat_names=feat_names, target=target, mg_col=mg_col, group_col="Location")


def build_features(df, feat_names, mg_col=None,
                   include_mg=False, mg_mode="raw", feature_mode="combined"):
    """Build the feature matrix from the harmonized dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized dataframe (from `normalize_colnames`).
    feat_names : list[str]
        Resolved base feature column names (from `resolve_feature_names`).
    mg_col : str, optional
        Mg column name (required if `include_mg=True`).
    include_mg : bool
        Whether to add Mg as a feature (ablation experiment in §3.2).
    mg_mode : {"raw", "log", "binary"}
        Mg encoding (only used if `include_mg=True`).
    feature_mode : {"raw_only", "engineered_only", "combined"}
        - raw_only: 8 base features only
        - engineered_only: 8 features (T, pH, EC, Na/K, Na+K, Cl/EC, B, SiO2)
        - combined: 11 features (default, used in main paper)

    Returns
    -------
    pd.DataFrame
        Feature matrix with appropriate columns.
    """
    X = df[feat_names].astype(float).copy()
    Na = X[feat_names[4]].values
    K = X[feat_names[3]].values
    EC = X[feat_names[2]].values
    Cl = X[feat_names[7]].values

    if feature_mode in ("engineered_only", "combined"):
        X["Na_K_ratio"] = np.clip((Na + 1e-6) / (K + 1e-6), 0.01, 100.0)
        X["Na_plus_K"] = Na + K
        X["Cl_over_EC"] = np.clip(Cl / (EC + 1e-6), 1e-4, 1.0)

    if feature_mode == "engineered_only":
        keep = [feat_names[0], feat_names[1], feat_names[2],
                "Na_K_ratio", "Na_plus_K", "Cl_over_EC",
                feat_names[5], feat_names[6]]
        X = X[[c for c in keep if c in X.columns]]

    if include_mg:
        if mg_col is None:
            raise ValueError("mg_col required when include_mg=True")
        mg = df[mg_col].astype(float).values
        if mg_mode == "log":
            X["Mg_log"] = np.log1p(mg)
        elif mg_mode == "binary":
            X["Mg_deep"] = (mg < 1.0).astype(float)
        else:
            X["Mg_raw"] = mg

    return X


def load_datasets(train_path, test_path):
    """Load and harmonize training and test CSVs.

    Returns
    -------
    dict with keys: train_n, test_n, feat_names, target, mg_col, group_col,
                    groups, y_train_raw, y_test_raw, X_train_raw, X_test_raw,
                    feature_names_ext
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_n = normalize_colnames(train_df)
    test_n = normalize_colnames(test_df)

    info = resolve_feature_names(train_n)
    feat_names = info["feat_names"]
    target = info["target"]
    mg_col = info["mg_col"]
    group_col = info["group_col"]

    groups = train_n[group_col].astype(str).values
    y_train_raw = train_n[target].astype(float).values
    y_test_raw = test_n[target].astype(float).values

    X_train_raw = build_features(train_n, feat_names)
    X_test_raw = build_features(test_n, feat_names)
    feature_names_ext = list(X_train_raw.columns)

    return dict(
        train_n=train_n,
        test_n=test_n,
        feat_names=feat_names,
        target=target,
        mg_col=mg_col,
        group_col=group_col,
        groups=groups,
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        feature_names_ext=feature_names_ext,
    )
