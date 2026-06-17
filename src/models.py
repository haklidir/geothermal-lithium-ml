"""
models.py — Model constructors and seeding utilities.

Defines the four regressors (DNN, Ridge, RF, XGB) plus ElasticNet, with
hyperparameters matching Section 3.3 and Appendix A.2 of the manuscript.
"""

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, ElasticNet
from xgboost import XGBRegressor

from .config import (
    SEED, DROPOUT, L2, LR, DNN_ARCHITECTURE,
    RF_PARAMS, XGB_PARAMS, RIDGE_ALPHAS, ENET_PARAMS,
)


def set_seeds(seed=SEED):
    """Set seeds for reproducibility across numpy, random, tf, hash."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass


def build_dnn(input_dim):
    """Compact feed-forward network: [input_dim] → 64 → 64 → 32 → 1.

    ELU activations, L2 regularization on weights, dropout between hidden
    layers, Adam optimizer with Huber loss. Trained on log1p(Li) with
    standardized inputs; output is inverted via expm1 at inference.

    Architecture and parameters match Section 3.3 (Appendix A.2.1).
    """
    h1, h2, h3 = DNN_ARCHITECTURE
    m = Sequential([
        Input(shape=(input_dim,)),
        Dense(h1, activation="elu", kernel_regularizer=regularizers.l2(L2)),
        Dropout(DROPOUT),
        Dense(h2, activation="elu", kernel_regularizer=regularizers.l2(L2)),
        Dropout(DROPOUT),
        Dense(h3, activation="elu", kernel_regularizer=regularizers.l2(L2)),
        Dense(1),
    ])
    m.compile(
        optimizer=Adam(learning_rate=LR, clipnorm=1.0),
        loss=Huber(delta=1.0),
        metrics=["mae"],
    )
    return m


def build_rf():
    """Random Forest (Appendix A.2.2)."""
    return RandomForestRegressor(**RF_PARAMS)


def build_xgb():
    """XGBoost (Appendix A.2.3)."""
    return XGBRegressor(**XGB_PARAMS)


def build_ridge():
    """Ridge regression with internal LOO-CV over candidate alphas (A.2.4)."""
    return RidgeCV(alphas=RIDGE_ALPHAS, scoring="neg_mean_squared_error")


def build_elasticnet():
    """ElasticNet baseline (supplementary)."""
    return ElasticNet(**ENET_PARAMS)


def inv_dnn(pred_std, mu, sigma):
    """Inverse-transform DNN output: standardized log1p → original mg/L scale.

    Clips negative predictions to zero (Li cannot be negative).
    """
    return np.maximum(np.expm1(np.maximum(pred_std * sigma + mu, 0.0)), 0.0)
