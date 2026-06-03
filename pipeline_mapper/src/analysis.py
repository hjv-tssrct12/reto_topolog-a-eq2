"""Phase 8: rigorous per-node statistical analysis."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.config import (
    CAT_ORDER,
    CI_BOOTSTRAP_N,
    CLIFF_DELTA_THRESHOLD,
    FEATURE_COLS,
    FDR_ALPHA,
    LENTE_COL,
    SEED,
)
from src.mapper_multiscale import Graph

log = logging.getLogger(__name__)


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size between two 1-D samples.

    Parameters
    ----------
    x, y : np.ndarray

    Returns
    -------
    float in [-1, 1]
    """
    dominance = 0
    for xi in x:
        dominance += int(np.sum(xi > y)) - int(np.sum(xi < y))
    return dominance / (len(x) * len(y))


def _bootstrap_ci(
    vals: np.ndarray,
    n_iter: int = CI_BOOTSTRAP_N,
    seed: int = SEED,
) -> Tuple[float, float]:
    """95% bootstrap confidence interval for the mean.

    Parameters
    ----------
    vals   : np.ndarray
    n_iter : bootstrap resamples.
    seed   : random seed.

    Returns
    -------
    (ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_iter)
    ])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def analyze_nodes(
    graph: Graph,
    lente_vals: np.ndarray,
    cat_vals: np.ndarray,
    df_work: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-node statistics with FDR correction and effect sizes.

    Parameters
    ----------
    graph       : kmapper Graph dict.
    lente_vals  : np.ndarray, shape (n,)  raw AF mg/day.
    cat_vals    : np.ndarray, shape (n,)  category labels.
    df_work     : pd.DataFrame with FEATURE_COLS and LENTE_COL columns.

    Returns
    -------
    df_analysis : pd.DataFrame, one row per node, sorted by AF_media.
    """
    all_af = lente_vals
    rows = []

    for node_id, members in graph["nodes"].items():
        members = list(members)
        node_af = lente_vals[members]
        cats    = cat_vals[members]
        feats   = df_work.iloc[members][FEATURE_COLS]

        n       = len(members)
        af_mean = float(node_af.mean())
        ci_lo, ci_hi = _bootstrap_ci(node_af)

        # Mann-Whitney: node AF vs rest
        rest_af = all_af[~np.isin(np.arange(len(all_af)), members)]
        if len(rest_af) > 0 and len(node_af) > 0:
            stat, p_mw = mannwhitneyu(node_af, rest_af, alternative="two-sided")
        else:
            stat, p_mw = np.nan, 1.0

        # Cliff's delta
        cd = _cliffs_delta(node_af, rest_af) if len(rest_af) > 0 else 0.0

        row = {
            "Nodo"        : node_id,
            "N"           : n,
            "AF_media"    : round(af_mean, 4),
            "CI_inf"      : round(ci_lo,   4),
            "CI_sup"      : round(ci_hi,   4),
            "p_mw"        : round(float(p_mw), 6),
            "cliffs_delta": round(cd, 4),
            "N_Bajo"      : int((cats == "Bajo").sum()),
            "N_Medio"     : int((cats == "Medio").sum()),
            "N_Alto"      : int((cats == "Alto").sum()),
            "pct_Bajo"    : round((cats == "Bajo").mean()  * 100, 1),
            "pct_Medio"   : round((cats == "Medio").mean() * 100, 1),
            "pct_Alto"    : round((cats == "Alto").mean()  * 100, 1),
        }
        for col in FEATURE_COLS:
            row[f"{col} (media)"] = round(feats[col].mean(), 4)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR correction
    if len(df) > 0:
        reject, p_fdr, _, _ = multipletests(df["p_mw"].fillna(1.0), method="fdr_bh")
        df["p_FDR"]            = np.round(p_fdr, 6)
        df["nodo_significativo"] = (
            (df["p_FDR"] < FDR_ALPHA) & (df["cliffs_delta"].abs() > CLIFF_DELTA_THRESHOLD)
        )
    else:
        df["p_FDR"]              = np.nan
        df["nodo_significativo"] = False

    # Dominant category
    def _dom(r: pd.Series) -> str:
        return max(CAT_ORDER, key=lambda c: r[f"N_{c}"])

    df["Cat_dominante"] = df.apply(_dom, axis=1)
    df = df.sort_values("AF_media").reset_index(drop=True)

    n_sig = int(df["nodo_significativo"].sum())
    log.info("Node analysis complete: %d nodes, %d significant (FDR<%.2f, |Δ|>%.2f)",
             len(df), n_sig, FDR_ALPHA, CLIFF_DELTA_THRESHOLD)
    return df
