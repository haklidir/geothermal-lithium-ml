# Geothermal Lithium ML

Code and data for the paper:

> **Tut Haklidir, F.S. and Haklidir, M. (2026).** Predicting Dissolved Lithium
> in Geothermal Fluids from Routine Hydrogeochemistry: A Machine Learning
> Framework with Calibrated Uncertainty Quantification.
> *Earth Science Informatics*, 19(9), 159. https://doi.org/10.1007/s12145-026-02207-6

This repository contains the curated dataset, the analysis code, and scripts
that reproduce every table and figure in the paper. The framework predicts
dissolved lithium concentrations in geothermal fluids from eleven routine
hydrogeochemical features using four regressors (DNN, Ridge, RF, XGB) under a
leakage-controlled GroupKFold cross-validation protocol, with empirical
prediction intervals sized from cross-validation residuals.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/haklidir/geothermal-lithium-ml.git
cd geothermal-lithium-ml
pip install -r requirements.txt

# 2. Reproduce the main results (Tables 3-5 + Figures 7, 8, 10, 11, 12)
python scripts/run_main_pipeline.py

# 3. Reproduce the LOGO-CV analysis (Section 4.3)
python scripts/run_logo_cv.py

# 4. Reproduce the bootstrap and LOO sensitivity (Section 4.2)
python scripts/run_bootstrap_loo.py

# 5. Reproduce the feature ablation (Section 3.2)
python scripts/run_feature_ablation.py

# 6. Regenerate the workflow schematic (Figure 9)
python scripts/make_fig9_workflow.py
```

All outputs are written under `results/tables/` (CSVs) and `results/figures/`
(PNGs).

## Repository layout

```
geothermal-lithium-ml/
├── README.md                              This file
├── LICENSE                                MIT
├── CITATION.cff                           Citation metadata
├── requirements.txt                       Pinned Python dependencies
│
├── data/
│   ├── training_dataset_lit.csv           74 training samples
│   └── testing_dataset_lit.csv            12 held-out test samples
│
├── src/                                   Importable Python package
│   ├── __init__.py
│   ├── config.py                          Hyperparameters and constants
│   ├── preprocessing.py                   Data loading and feature engineering
│   ├── models.py                          DNN, Ridge, RF, XGB constructors
│   ├── pipeline.py                        Main GroupKFold pipeline
│   ├── logo_cv.py                         Leave-one-group-out CV
│   ├── evaluation.py                      Metrics, prediction intervals, bootstrap
│   └── plotting.py                        Figure generation
│
├── scripts/                               Runnable entry points
│   ├── run_main_pipeline.py               Tables 3–5, Figures 7, 8, 10–12
│   ├── run_logo_cv.py                     LOGO-CV across 26 location groups
│   ├── run_bootstrap_loo.py               DNN vs Ridge bootstrap, LOO sensitivity
│   ├── run_feature_ablation.py            Raw vs engineered vs combined features
│   └── make_fig9_workflow.py              Figure 9 schematic via Graphviz
│
├── notebooks/
│   └── LitPred_FINAL_RERUN.ipynb          Original end-to-end Colab notebook
│
├── results/
│   ├── tables/                            Generated CSV outputs
│   └── figures/                           Generated PNG outputs
│
└── docs/
    └── reproducibility.md                 Notes on reproducibility
```

## Methodology summary

- **Dataset.** 86 thermal springs and wells compiled from published sources
  across Türkiye, predominantly Western Anatolia (74 train, 12 test).
- **Features.** 8 base hydrogeochemical features (T, pH, EC, Na, K, Cl, B,
  SiO₂) plus 3 engineered ratios (Na/K, Na+K, Cl/EC).
- **Models.** A compact neural network ([64, 64, 32] units, ELU, dropout 0.12,
  L2 1×10⁻⁴, Adam at 5×10⁻⁴, early stopping with patience 60), Ridge
  regression (RidgeCV over five alphas), Random Forest (900 trees,
  max_depth 20, min_samples_leaf 3, max_features="sqrt"), and XGBoost
  (900 boosting rounds, max_depth 5, learning_rate 0.05, subsample 0.8,
  colsample_bytree 0.8, reg_lambda 2.0, reg_alpha 0.1).
- **Validation.** Five-fold GroupKFold over 26 location groups to prevent
  spatial leakage. The per-sample median over the five fold models is used as
  the ensemble prediction.
- **Uncertainty.** Two-sided empirical prediction intervals at the 90%
  nominal level, sized from pooled cross-validation absolute residuals.
  Inspired by split-conformal prediction but not strictly conformal — see
  Section 3.5 of the paper.
- **Generalization.** A leave-one-group-out cross-validation (LOGO-CV)
  across all 26 location groups is reported as the most defensible
  single estimate of out-of-domain performance.

## Key headline numbers (paper-reported)

| Metric                          | DNN   | Ridge | RF    | XGB   |
|---------------------------------|-------|-------|-------|-------|
| Test RMSE (mg/L)                | 1.18  | 1.65  | 3.39  | 1.87  |
| Test R²                         | 0.95  | 0.91  | 0.62  | 0.88  |
| 90% Empirical coverage          | 100%  | 91.7% | 91.7% | 91.7% |
| LOGO-CV R²                      | 0.67  | −226  | 0.44  | 0.54  |
| Paired bootstrap (vs Ridge), p  | —     | 0.42  | —     | —     |

The Ridge LOGO-CV collapse (R² = −226, RMSE 59.2 mg/L) is driven primarily
by the Tuzla T1 fold; see Section 4.3 of the paper for discussion.

## Reproducibility

All random number generators (NumPy, Python `random`, TensorFlow) are seeded
to 42, hyperparameters are pinned in `src/config.py`, and software versions
are listed in `requirements.txt`. Minor non-determinism in TensorFlow can
result in R² shifts at the third decimal place across hardware.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`).

## License

MIT — see `LICENSE`.

## Contact

- Mehmet Haklidir — `mehmet.haklidir@tubitak.gov.tr`
- Füsun S. Tut Haklidir — `fusun.tut@bilgi.edu.tr`
