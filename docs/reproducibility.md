# Reproducibility Notes

This document records the random seeds, hyperparameters, and software
versions used to produce the manuscript results, so that anyone running the
code on a fresh environment can verify the headline numbers.

## Random seeds

All RNGs are seeded to **42**:

- Python `random.seed(42)`
- NumPy `np.random.seed(42)`
- TensorFlow `tf.random.set_seed(42)`
- `PYTHONHASHSEED=42`
- TensorFlow op determinism enabled where supported

See `src/models.py::set_seeds()`.

## Hyperparameters

All hyperparameters are pinned in `src/config.py` and match Appendix A.2 of
the manuscript exactly.

### DNN (Appendix A.2.1)

```
Architecture: [64, 64, 32] hidden units, ELU activations
Dropout: 0.12 between hidden layers
L2 weight decay: 1e-4
Optimizer: Adam, learning_rate = 5e-4, clipnorm = 1.0
Loss: Huber (delta = 1.0)
Batch size: 32
Max epochs: 800
Early stopping: patience = 60, restore_best_weights = True
ReduceLROnPlateau: factor = 0.5, patience = 30, min_lr = 1e-6
Target transform: log1p, then standardize per fold
Output inverse: clip(expm1(unstd), 0, ∞)
Nominal parameters: 7,041
```

### Random Forest (Appendix A.2.2)

```
n_estimators = 900
max_depth = 20
min_samples_leaf = 3
max_features = "sqrt"
random_state = 42
```

### XGBoost (Appendix A.2.3)

```
n_estimators = 900
max_depth = 5
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
reg_lambda = 2.0
reg_alpha = 0.1
tree_method = "hist"
random_state = 42
```

### Ridge (Appendix A.2.4)

```
RidgeCV over alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
scoring = "neg_mean_squared_error"
Inputs: StandardScaler-transformed
Target: log1p(Li), inverse via expm1, clipped to [0, ∞)
```

## Software versions

See `requirements.txt` for pinned ranges. The reported numbers were produced
under:

- Python 3.10
- numpy 1.26
- pandas 2.0
- scikit-learn 1.3
- xgboost 2.0
- tensorflow 2.15
- shap 0.43
- scipy 1.10
- matplotlib 3.7

## Expected runtime

Approximate wall-clock times on a 2024 laptop CPU (no GPU):

- `run_main_pipeline.py`        ~3–5 min (DNN dominates)
- `run_logo_cv.py`              ~25–40 min (26 LOGO folds × 4 models)
- `run_bootstrap_loo.py`        ~3–5 min
- `run_feature_ablation.py`     ~10–15 min (DNN × 3 configs × 5 folds)
- `make_fig9_workflow.py`       <5 s

## Determinism caveats

Even with full seeding, TensorFlow can produce small numerical differences
across hardware (CPU vs GPU, different CPU instruction sets, TF version).
This typically affects R² values at the third decimal place and does not
change the qualitative conclusions of the paper. The headline metrics
(R² ≈ 0.95 for DNN test set, R² ≈ 0.67 for DNN LOGO-CV) should reproduce
within ±0.01 on any modern x86_64 system.

## Verifying a successful run

After running `run_main_pipeline.py`, check `results/tables/TABLE3_test_metrics.csv`:

```
Model,RMSE (mg/L),MAE (mg/L),R²,MAPE (%),Empirical Coverage,Mean Width (mg/L)
DNN,1.177,0.918,0.954,17.3,100.0%,6.026
Ridge,1.652,1.032,0.909,15.7,91.7%,5.760
RF,3.391,1.546,0.615,16.8,91.7%,4.556
XGB,1.867,1.000,0.883,14.8,91.7%,4.814
```

If your numbers differ at the third decimal place, this is expected
hardware-level non-determinism. If they differ in the first or second
decimal, please open a GitHub issue with your environment details.
