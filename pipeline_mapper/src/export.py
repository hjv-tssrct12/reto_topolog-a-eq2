"""Phase 10: multi-sheet Excel export."""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import FEATURE_COLS, LENTE_COL, OUTPUTS_DIR

log = logging.getLogger(__name__)


def _pip_freeze_relevant() -> str:
    """Capture versions of key packages via pip freeze."""
    relevant = {
        "kmapper", "umap-learn", "ripser", "persim",
        "scikit-learn", "scipy", "statsmodels", "pandas", "numpy",
        "matplotlib", "plotly", "networkx", "openpyxl",
    }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=20,
        )
        lines = [
            ln for ln in result.stdout.splitlines()
            if any(pkg in ln.lower() for pkg in relevant)
        ]
        return "\n".join(lines)
    except Exception:
        return "pip freeze failed"


def _executive_summary(df_analysis: pd.DataFrame) -> pd.DataFrame:
    """Build clinical interpretation table comparing Alto vs Bajo nodes."""
    rows = []
    for grupo in ["Alto", "Bajo", "Medio"]:
        sub = df_analysis[df_analysis["Cat_dominante"] == grupo]
        if sub.empty:
            continue
        row = {"Grupo": f"{grupo} AF", "n_nodos": len(sub),
               "AF_media_mg_dia": round(sub["AF_media"].mean(), 4)}
        for col in FEATURE_COLS:
            col_key = f"{col} (media)"
            if col_key in sub.columns:
                row[col] = round(sub[col_key].mean(), 4)
        row["CI_inf_AF"] = round(sub["CI_inf"].mean(), 4)
        row["CI_sup_AF"] = round(sub["CI_sup"].mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_gs_hyperparams(gs_config: Dict) -> pd.DataFrame:
    """Build a summary table of the grid search scoring weights and winner."""
    rows = [
        {"Concepto": "── Configuración ganadora ──",      "Valor": ""},
        {"Concepto": "n_cubes (grid search)",              "Valor": gs_config["n_cubes"]},
        {"Concepto": "overlap (grid search)",              "Valor": gs_config["overlap"]},
        {"Concepto": "composite_score (sin penalizar)",    "Valor": gs_config.get("composite_score", gs_config.get("score", "N/A"))},
        {"Concepto": "size_penalty",                       "Valor": gs_config.get("size_penalty", "N/A")},
        {"Concepto": "penalised_score (score final)",      "Valor": gs_config.get("penalised_score", gs_config.get("score", "N/A"))},
        {"Concepto": "", "Valor": ""},
        {"Concepto": "── Pesos del score compuesto ──",    "Valor": ""},
        {"Concepto": "lcc_fraction (conectividad)",        "Valor": 0.25},
        {"Concepto": "cat_separation (separación AF)",     "Valor": 0.25},
        {"Concepto": "node_cohesion (cohesión intra-nodo)","Valor": 0.20},
        {"Concepto": "overlap_ratio (riqueza overlap)",    "Valor": 0.15},
        {"Concepto": "n_cycles_norm (ciclos topológicos)", "Valor": 0.15},
        {"Concepto": "", "Valor": ""},
        {"Concepto": "── Penalización por tamaño ──",      "Valor": ""},
        {"Concepto": "fórmula",                            "Valor": "score_final = composite × 1/(1+exp(k×(n_nodes−target)))"},
        {"Concepto": "penalty_target (nodos)",             "Valor": 250},
        {"Concepto": "penalty_k",                          "Valor": 0.03},
        {"Concepto": "efecto: n≤180 nodos",                "Valor": "penalización < 13%"},
        {"Concepto": "efecto: n=250 nodos",                "Valor": "penalización = 50%"},
        {"Concepto": "efecto: n≥300 nodos",                "Valor": "penalización > 80%"},
    ]
    return pd.DataFrame(rows)


def export_excel(
    df_analysis: pd.DataFrame,
    sweep_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    perturbation_stats: Dict,
    h1_summary: Dict,
    chosen_config: Dict,
    umap_results: pd.DataFrame,
    ml_reg_results: Optional[pd.DataFrame] = None,
    ml_clf_results: Optional[pd.DataFrame] = None,
    gs_results_df: Optional[pd.DataFrame] = None,   # ← grid search ranking
    gs_config: Optional[Dict] = None,               # ← grid search winner
    df_feature_importance: Optional[pd.DataFrame] = None,  # ← Phase 9b
) -> Path:
    """Write the multi-sheet Excel report.

    Sheets
    ------
    1.  nodos_estadisticas      — per-node statistics (Phase 8)
    2.  barrido_multiescala     — multiscale sweep (Phase 5a)
    3.  estabilidad_bootstrap   — bootstrap stability (Phase 6a)
    4.  estabilidad_perturbacion— perturbation stability (Phase 6b)
    5.  homologia_persistente   — persistent homology summary (Phase 3)
    6.  hiperparametros         — all hyperparameters + pip versions
    7.  interpretacion_clinica  — executive summary by AF group
    8.  ml_regresion            — ML regression grid search (Phase 9)
    9.  ml_clasificacion        — ML classification grid search (Phase 9)
    10. mapper_grid_search      — Mapper grid search ranking (Phase 5b) ← NEW
    11. mapper_gs_config        — Grid search winner + weight explanation ← NEW
    12. feature_importance      — Feature importance + significance (Phase 9b) ← NEW

    Parameters
    ----------
    df_analysis        : per-node statistics.
    sweep_df           : multiscale sweep results.
    bootstrap_df       : patient-level bootstrap probabilities.
    perturbation_stats : dict from stability.perturbation_stability.
    h1_summary         : dict from topological_exploration.
    chosen_config      : dict with n_cubes, overlap (multiscale sweep winner).
    umap_results       : UMAP grid search results DataFrame.
    ml_reg_results     : ML regression results (optional).
    ml_clf_results     : ML classification results (optional).
    gs_results_df      : Mapper grid search full ranking (optional).
    gs_config          : Mapper grid search best config dict (optional).

    Returns
    -------
    output_path : Path to the written Excel file.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    output_path = OUTPUTS_DIR / "reporte_tda_final.xlsx"

    # Sheet 6 — hyperparameters
    hp_rows = [
        {"Parámetro": "seed",                   "Valor": 42},
        {"Parámetro": "── Multiscale sweep ──", "Valor": ""},
        {"Parámetro": "n_cubes_sweep",          "Valor": chosen_config["n_cubes"]},
        {"Parámetro": "overlap_sweep",          "Valor": chosen_config["overlap"]},
        {"Parámetro": "── Grid search ──",      "Valor": ""},
        {"Parámetro": "n_cubes_gs",
         "Valor": gs_config["n_cubes"] if gs_config else "N/A"},
        {"Parámetro": "overlap_gs",
         "Valor": gs_config["overlap"] if gs_config else "N/A"},
        {"Parámetro": "gs_composite_score",
         "Valor": gs_config.get("composite_score", gs_config.get("score", "N/A")) if gs_config else "N/A"},
        {"Parámetro": "gs_size_penalty",
         "Valor": gs_config.get("size_penalty", "N/A") if gs_config else "N/A"},
        {"Parámetro": "gs_penalised_score",
         "Valor": gs_config.get("penalised_score", gs_config.get("score", "N/A")) if gs_config else "N/A"},
        {"Parámetro": "── DBSCAN / Stability ──", "Valor": ""},
        {"Parámetro": "dbscan_adaptive",        "Valor": "True"},
        {"Parámetro": "bootstrap_n",            "Valor": 50},
        {"Parámetro": "perturbation_n",         "Valor": 20},
        {"Parámetro": "perturbation_sigma",     "Valor": 0.01},
        {"Parámetro": "── Analysis ──",         "Valor": ""},
        {"Parámetro": "fdr_alpha",              "Valor": 0.05},
        {"Parámetro": "cliff_delta_threshold",  "Valor": 0.33},
        {"Parámetro": "pip_versions",           "Valor": _pip_freeze_relevant()},
    ]
    df_hp = pd.DataFrame(hp_rows)

    # Sheet 5 — persistent homology
    df_h1 = pd.DataFrame([h1_summary])

    # Sheet 4 — perturbation
    df_pert = pd.DataFrame([{
        "similitud_media": perturbation_stats["mean"],
        "similitud_std"  : perturbation_stats["std"],
    }])

    # Sheet 7 — executive summary
    df_exec = _executive_summary(df_analysis)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # ── Core sheets (unchanged) ────────────────────────────────────────
        df_analysis.to_excel(
            writer, sheet_name="nodos_estadisticas",      index=False)
        sweep_df.to_excel(
            writer, sheet_name="barrido_multiescala",     index=False)
        bootstrap_df.to_excel(
            writer, sheet_name="estabilidad_bootstrap",   index=False)
        df_pert.to_excel(
            writer, sheet_name="estabilidad_perturbacion",index=False)
        df_h1.to_excel(
            writer, sheet_name="homologia_persistente",   index=False)
        df_hp.to_excel(
            writer, sheet_name="hiperparametros",         index=False)
        df_exec.to_excel(
            writer, sheet_name="interpretacion_clinica",  index=False)

        # ── ML sheets ─────────────────────────────────────────────────────
        if ml_reg_results is not None:
            ml_reg_results.to_excel(
                writer, sheet_name="ml_regresion",        index=False)
        if ml_clf_results is not None:
            ml_clf_results.to_excel(
                writer, sheet_name="ml_clasificacion",    index=False)

        # ── Mapper grid search sheets (new) ───────────────────────────────
        if gs_results_df is not None:
            # Full ranking — all configs sorted by composite_score desc
            gs_results_df.to_excel(
                writer, sheet_name="mapper_grid_search",  index=False)

        if gs_config is not None:
            # Winner summary + weight explanation
            df_gs_hp = _build_gs_hyperparams(gs_config)
            df_gs_hp.to_excel(
                writer, sheet_name="mapper_gs_config",    index=False)

        # ── Feature importance sheet (new) ──────────────────────────────
        if df_feature_importance is not None:
            df_feature_importance.to_excel(
                writer, sheet_name="feature_importance", index=False)

    log.info("Excel report written: %s  (%d sheets)",
             output_path,
             9 + (gs_results_df is not None) + (gs_config is not None)
             + (ml_reg_results is not None) + (ml_clf_results is not None))
    return output_path