"""
config.py — Hyperparameters and constants for the geothermal Li prediction framework.

All values match the manuscript Appendix A.2.
"""

# Global seed and CV settings
SEED = 42
N_SPLITS = 5

# DNN hyperparameters (Appendix A.2.1)
BATCH_SIZE = 32
EPOCHS = 800
PATIENCE = 60
LR = 5e-4
L2 = 1e-4
DROPOUT = 0.12
DNN_ARCHITECTURE = [64, 64, 32]  # Hidden layer sizes

# Prediction interval coverage (90% nominal)
CP_COVERAGE = 0.90

# Random Forest hyperparameters (Appendix A.2.2)
RF_PARAMS = dict(
    n_estimators=900,
    max_depth=20,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)

# XGBoost hyperparameters (Appendix A.2.3)
XGB_PARAMS = dict(
    n_estimators=900,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.1,
    random_state=SEED,
    tree_method="hist",
)

# Ridge regression candidate alphas (Appendix A.2.4)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# ElasticNet (reported in supplementary; collapses on this dataset)
ENET_PARAMS = dict(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=SEED)

# Model list used throughout
ALL_MODELS = ["DNN", "Ridge", "RF", "XGB", "ElasticNet"]
HEADLINE_MODELS = ["DNN", "Ridge", "RF", "XGB"]  # Reported in tables
