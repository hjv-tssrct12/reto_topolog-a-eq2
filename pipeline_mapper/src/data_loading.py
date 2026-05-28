"""Phase 1-2: data loading, cleaning, and feature scaling."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.config import (
    BD_PATH, LENTE_COL, CAT_COL, CAT_ORDER, FEATURE_COLS,
)

log = logging.getLogger(__name__)


def load_and_clean() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load the clean dataset and apply basic imputation.

    Parameters
    ----------
    (none — path taken from config)

    Returns
    -------
    df_work : pd.DataFrame
        Rows with valid LENTE_COL, features imputed with column median.
    lente_vals : np.ndarray, shape (n,)
        Raw mg/day AF values.
    cat_vals : np.ndarray, shape (n,)  dtype=str
        Category label per row: "Bajo" / "Medio" / "Alto".
    cat_enc : np.ndarray, shape (n,)  dtype=float
        Encoded category: 0 / 1 / 2.
    """
    log.info("Loading %s", BD_PATH)
    df = pd.read_excel(BD_PATH)
    log.info("Raw shape: %s", df.shape)

    for col in [LENTE_COL, CAT_COL] + FEATURE_COLS:
        assert col in df.columns, f"Column not found: {col}"

    df_work = df[[LENTE_COL, CAT_COL] + FEATURE_COLS].copy()

    n_before = len(df_work)
    df_work = df_work.dropna(subset=[LENTE_COL]).reset_index(drop=True)
    log.info("Rows dropped (NaN in lens): %d", n_before - len(df_work))
    log.info("Rows for Mapper: %d", len(df_work))

    for col in FEATURE_COLS:
        if df_work[col].isna().any():
            med = df_work[col].median()
            df_work[col] = df_work[col].fillna(med)
            log.info("Imputed '%s' with median=%.4f", col, med)

    assert df_work[FEATURE_COLS].isna().sum().sum() == 0, "NaN remaining in features"

    lente_vals = df_work[LENTE_COL].values
    cat_vals   = df_work[CAT_COL].values
    cat_enc    = np.array([CAT_ORDER.index(c) for c in cat_vals], dtype=float)

    for cat in CAT_ORDER:
        n = (cat_vals == cat).sum()
        log.info("  %-5s : n=%d  (%.1f%%)", cat, n, n / len(cat_vals) * 100)

    return df_work, lente_vals, cat_vals, cat_enc


def prepare_features(df_work: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise feature matrix with MinMaxScaler and RobustScaler.

    Parameters
    ----------
    df_work : pd.DataFrame
        Cleaned dataframe (output of load_and_clean).

    Returns
    -------
    X_minmax : np.ndarray, shape (n, p)
        Features scaled to [0, 1].
    X_robust : np.ndarray, shape (n, p)
        Features scaled with RobustScaler (median / IQR), then clipped [0, 1].
    """
    raw = df_work[FEATURE_COLS].values

    X_minmax = MinMaxScaler().fit_transform(raw)

    robust = RobustScaler().fit_transform(raw)
    X_robust = MinMaxScaler().fit_transform(robust)   # bring back to [0,1] for UMAP

    log.info("Feature matrix shape: %s | scalers: MinMax, Robust", raw.shape)
    return X_minmax, X_robust
