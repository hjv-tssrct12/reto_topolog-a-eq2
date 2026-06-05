"""Phase 9b: Feature importance and statistical significance analysis.

Three complementary approaches:

1. **Kruskal-Wallis + FDR** — non-parametric test per feature across AF
   categories (Bajo / Medio / Alto).  Gives formal p-values and effect sizes
   (eta-squared) for each variable.

2. **Random Forest feature importances** — mean decrease in impurity from the
   best RF regressor and classifier, averaged and ranked.  Gives a magnitude
   signal complementary to the p-values.

3. **Permutation importances** — model-agnostic, avoids the high-cardinality
   bias of impurity-based importances.  Run on both best models.

All results are merged into a single ranked DataFrame for easy reporting.

Public API
----------
feature_importance_analysis(X, y_cont, y_cat, feature_names,
                             best_reg, best_clf) -> pd.DataFrame
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, KFold
from statsmodels.stats.multitest import multipletests

from src.config import CV_FOLDS, FEATURE_COLS, OUTPUTS_DIR, SEED

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper — eta-squared effect size for Kruskal-Wallis
# ---------------------------------------------------------------------------

def _eta_squared_kw(H: float, n: int, k: int) -> float:
    """Eta-squared from Kruskal-Wallis H statistic.

    eta² = (H - k + 1) / (n - k)

    Interpretation: 0.01 small | 0.06 medium | 0.14 large
    """
    if n <= k:
        return 0.0
    return max(0.0, (H - k + 1) / (n - k))


# ---------------------------------------------------------------------------
# 1. Kruskal-Wallis per feature
# ---------------------------------------------------------------------------

def _kruskal_wallis(
    X: np.ndarray,
    y_cat: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Kruskal-Wallis H-test for each feature across AF categories.

    Returns DataFrame with columns:
        feature, H_stat, p_value, p_fdr, eta_squared, significativa
    """
    categories = np.unique(y_cat)
    n          = len(y_cat)
    k          = len(categories)
    rows       = []

    for i, feat in enumerate(feature_names):
        groups = [X[y_cat == cat, i] for cat in categories]
        # Need at least 2 values per group
        if any(len(g) < 2 for g in groups):
            rows.append({"feature": feat, "H_stat": np.nan, "p_value": 1.0})
            continue
        try:
            H, p = kruskal(*groups)
        except Exception:
            H, p = np.nan, 1.0
        rows.append({"feature": feat, "H_stat": round(float(H), 4),
                     "p_value": round(float(p), 6)})

    df = pd.DataFrame(rows)

    # FDR correction
    reject, p_fdr, _, _ = multipletests(
        df["p_value"].fillna(1.0), method="fdr_bh"
    )
    df["p_fdr"]       = np.round(p_fdr, 6)
    df["eta_squared"] = df.apply(
        lambda r: round(_eta_squared_kw(r["H_stat"] if not np.isnan(r["H_stat"]) else 0,
                                        n, k), 4),
        axis=1,
    )
    df["kw_significativa"] = reject

    log.info(
        "Kruskal-Wallis: %d / %d features significant (FDR<0.05)",
        int(reject.sum()), len(df),
    )
    return df


# ---------------------------------------------------------------------------
# 2. RF feature importances (impurity-based)
# ---------------------------------------------------------------------------

def _rf_importances(
    X: np.ndarray,
    y_cont: np.ndarray,
    y_cat:  np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Fit RF regressor and classifier with best known params, extract importances."""

    # Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1,
    )
    rf_reg.fit(X, y_cont)
    imp_reg = rf_reg.feature_importances_

    # Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1,
    )
    rf_clf.fit(X, y_cat)
    imp_clf = rf_clf.feature_importances_

    df = pd.DataFrame({
        "feature"          : feature_names,
        "rf_imp_reg"       : np.round(imp_reg, 4),
        "rf_imp_clf"       : np.round(imp_clf, 4),
        "rf_imp_mean"      : np.round((imp_reg + imp_clf) / 2, 4),
    })

    log.info("RF importances computed (reg + clf)")
    return df


# ---------------------------------------------------------------------------
# 3. Permutation importances (model-agnostic, CV-based)
# ---------------------------------------------------------------------------

def _permutation_importances(
    X: np.ndarray,
    y_cont: np.ndarray,
    y_cat:  np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance on held-out fold from RF reg and clf."""

    rng = np.random.default_rng(SEED)

    # Use one fold for speed
    kf  = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    train_r, val_r = next(iter(kf.split(X, y_cont)))
    train_c, val_c = next(iter(skf.split(X, y_cat)))

    # Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1,
    )
    rf_reg.fit(X[train_r], y_cont[train_r])
    perm_reg = permutation_importance(
        rf_reg, X[val_r], y_cont[val_r],
        n_repeats=n_repeats, random_state=SEED, scoring="r2",
    )

    # Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1,
    )
    rf_clf.fit(X[train_c], y_cat[train_c])
    perm_clf = permutation_importance(
        rf_clf, X[val_c], y_cat[val_c],
        n_repeats=n_repeats, random_state=SEED, scoring="f1_macro",
    )

    df = pd.DataFrame({
        "feature"         : feature_names,
        "perm_imp_reg"    : np.round(perm_reg.importances_mean,  4),
        "perm_imp_reg_std": np.round(perm_reg.importances_std,   4),
        "perm_imp_clf"    : np.round(perm_clf.importances_mean,  4),
        "perm_imp_clf_std": np.round(perm_clf.importances_std,   4),
        "perm_imp_mean"   : np.round(
            (perm_reg.importances_mean + perm_clf.importances_mean) / 2, 4
        ),
    })

    log.info("Permutation importances computed")
    return df


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def feature_importance_analysis(
    X:             np.ndarray,
    y_cont:        np.ndarray,
    y_cat:         np.ndarray,
    feature_names: List[str] = None,
) -> pd.DataFrame:
    """Full feature importance analysis combining three methods.

    Parameters
    ----------
    X             : np.ndarray, shape (n, p) — scaled features.
    y_cont        : np.ndarray, shape (n,)   — AF continuo (mg/día).
    y_cat         : np.ndarray, shape (n,)   — AF categórico (str).
    feature_names : list of str (default: FEATURE_COLS from config).

    Returns
    -------
    df_importance : pd.DataFrame ranked by composite_rank, with columns:
        feature, H_stat, p_value, p_fdr, eta_squared, kw_significativa,
        rf_imp_reg, rf_imp_clf, rf_imp_mean,
        perm_imp_reg, perm_imp_clf, perm_imp_mean,
        composite_rank
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if feature_names is None:
        feature_names = FEATURE_COLS

    log.info("Feature importance analysis — %d features", len(feature_names))

    # ── 1. Kruskal-Wallis ─────────────────────────────────────────────────
    log.info("[Feature analysis 1/3] Kruskal-Wallis + FDR ...")
    df_kw   = _kruskal_wallis(X, y_cat, feature_names)

    # ── 2. RF importances ─────────────────────────────────────────────────
    log.info("[Feature analysis 2/3] RF feature importances ...")
    df_rf   = _rf_importances(X, y_cont, y_cat, feature_names)

    # ── 3. Permutation importances ─────────────────────────────────────────
    log.info("[Feature analysis 3/3] Permutation importances ...")
    df_perm = _permutation_importances(X, y_cont, y_cat, feature_names)

    # ── Merge ──────────────────────────────────────────────────────────────
    df = df_kw.merge(df_rf,   on="feature").merge(df_perm, on="feature")

    # ── Composite rank — average rank across three importance signals ──────
    # Each signal ranked 1=most important; lower composite = more important
    df["rank_kw"]   = df["eta_squared"].rank(ascending=False).astype(int)
    df["rank_rf"]   = df["rf_imp_mean"].rank(ascending=False).astype(int)
    df["rank_perm"] = df["perm_imp_mean"].rank(ascending=False).astype(int)
    df["composite_rank"] = (
        (df["rank_kw"] + df["rank_rf"] + df["rank_perm"]) / 3
    ).round(2)

    df.sort_values("composite_rank", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["rank_kw", "rank_rf", "rank_perm"], inplace=True)

    # ── Save CSV ────────────────────────────────────────────────────────────
    df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    log.info("Feature importance saved: feature_importance.csv")

    # ── Log top features ────────────────────────────────────────────────────
    log.info("Top 5 features by composite rank:")
    for _, row in df.head(5).iterrows():
        log.info(
            "  %-25s  eta²=%.3f  rf_mean=%.3f  perm_mean=%.3f  "
            "p_fdr=%.4f  sig=%s",
            row["feature"],
            row["eta_squared"],
            row["rf_imp_mean"],
            row["perm_imp_mean"],
            row["p_fdr"],
            "✓" if row["kw_significativa"] else "✗",
        )

    return df