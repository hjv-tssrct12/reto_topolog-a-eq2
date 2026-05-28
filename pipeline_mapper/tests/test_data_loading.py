"""Tests for data_loading module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import load_and_clean, prepare_features
from src.config import FEATURE_COLS, CAT_ORDER


def test_load_and_clean_shape():
    df_work, lente_vals, cat_vals, cat_enc = load_and_clean()
    assert len(df_work) > 0, "Dataset should not be empty"
    assert len(lente_vals) == len(df_work)
    assert len(cat_vals) == len(df_work)
    assert len(cat_enc) == len(df_work)


def test_no_nan_after_clean():
    df_work, _, _, _ = load_and_clean()
    assert df_work[FEATURE_COLS].isna().sum().sum() == 0


def test_cat_labels_valid():
    _, _, cat_vals, cat_enc = load_and_clean()
    assert set(cat_vals).issubset(set(CAT_ORDER))
    assert set(cat_enc).issubset({0.0, 1.0, 2.0})


def test_prepare_features_range():
    df_work, _, _, _ = load_and_clean()
    X_minmax, X_robust = prepare_features(df_work)
    assert X_minmax.shape[1] == len(FEATURE_COLS)
    assert X_robust.shape[1] == len(FEATURE_COLS)
    assert X_minmax.min() >= -1e-9 and X_minmax.max() <= 1.0 + 1e-9
    assert X_robust.min() >= -1e-9 and X_robust.max() <= 1.0 + 1e-9


def test_prepare_features_no_nan():
    df_work, _, _, _ = load_and_clean()
    X_minmax, X_robust = prepare_features(df_work)
    assert not np.isnan(X_minmax).any()
    assert not np.isnan(X_robust).any()
