"""Phase 4: UMAP dimensionality reduction with grid search and scaler comparison."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import umap
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    OUTPUTS_DIR,
    SEED,
    UMAP_MIN_DIST_CANDIDATES,
    UMAP_N_NEIGHBORS_CANDIDATES,
    UMAP_TRUST_N_NEIGHBORS,
)

log = logging.getLogger(__name__)


def _continuity(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int) -> float:
    """Continuity: how well high-D neighborhoods are preserved in low-D space.

    Computed as trustworthiness with X_high and X_low roles swapped.

    Parameters
    ----------
    X_high : np.ndarray
    X_low  : np.ndarray
    n_neighbors : int

    Returns
    -------
    float in [0, 1]
    """
    return float(trustworthiness(X_low, X_high, n_neighbors=n_neighbors))


def _spearman_distance_correlation(X_high: np.ndarray, X_low: np.ndarray) -> float:
    """Spearman correlation between pairwise distances in high-D and low-D space.

    Parameters
    ----------
    X_high : np.ndarray, shape (n, p_high)
    X_low  : np.ndarray, shape (n, 2)

    Returns
    -------
    float — Spearman rho in [-1, 1]
    """
    d_high = pdist(X_high, metric="euclidean")
    d_low  = pdist(X_low,  metric="euclidean")
    rho, _ = spearmanr(d_high, d_low)
    return float(rho)


def _fit_umap(
    X: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray:
    """Fit a single UMAP and return the 2-D embedding."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(X)


def umap_grid_search(
    X_minmax: np.ndarray,
    X_robust: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame, str]:
    """Grid-search over n_neighbors × min_dist for both scalers.

    Selects the embedding with the highest combined score
    (trustworthiness + continuity) / 2.

    Parameters
    ----------
    X_minmax : np.ndarray, shape (n, p)
    X_robust : np.ndarray, shape (n, p)

    Returns
    -------
    best_embedding : np.ndarray, shape (n, 2)  — normalised to [0, 1]
    results_df     : pd.DataFrame              — all 18 configs + metrics
    best_scaler    : str                       — "minmax" or "robust"
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    scalers = {"minmax": X_minmax, "robust": X_robust}
    rows: List[Dict] = []
    best_score = -np.inf
    best_embedding: np.ndarray = np.zeros((X_minmax.shape[0], 2))
    best_scaler = "minmax"

    total = len(scalers) * len(UMAP_N_NEIGHBORS_CANDIDATES) * len(UMAP_MIN_DIST_CANDIDATES)
    done = 0

    for scaler_name, X in scalers.items():
        for nn in UMAP_N_NEIGHBORS_CANDIDATES:
            for md in UMAP_MIN_DIST_CANDIDATES:
                done += 1
                log.info("[UMAP %d/%d] scaler=%s  n_neighbors=%d  min_dist=%.1f",
                         done, total, scaler_name, nn, md)

                emb = _fit_umap(X, nn, md, SEED)

                trust = float(trustworthiness(X, emb, n_neighbors=UMAP_TRUST_N_NEIGHBORS))
                cont  = _continuity(X, emb, n_neighbors=UMAP_TRUST_N_NEIGHBORS)
                rho   = _spearman_distance_correlation(X, emb)
                score = (trust + cont) / 2.0

                rows.append({
                    "scaler"      : scaler_name,
                    "n_neighbors" : nn,
                    "min_dist"    : md,
                    "trustworthiness": round(trust, 4),
                    "continuity"  : round(cont, 4),
                    "spearman_rho": round(rho, 4),
                    "combined_score": round(score, 4),
                })

                if score > best_score:
                    best_score    = score
                    best_embedding = emb
                    best_scaler   = scaler_name

    results_df = pd.DataFrame(rows).sort_values("combined_score", ascending=False)
    results_df.to_csv(OUTPUTS_DIR / "umap_grid_search.csv", index=False)
    log.info("UMAP grid search saved: umap_grid_search.csv")

    best_row = results_df.iloc[0]
    log.info(
        "Best UMAP: scaler=%s  n_neighbors=%d  min_dist=%.1f  score=%.4f",
        best_row["scaler"], best_row["n_neighbors"], best_row["min_dist"],
        best_row["combined_score"],
    )

    norm_embedding = MinMaxScaler().fit_transform(best_embedding)
    return norm_embedding, results_df, best_scaler
