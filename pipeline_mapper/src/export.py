"""Phase 10: multi-sheet Excel export."""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict

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


def export_excel(
    df_analysis: pd.DataFrame,
    sweep_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    perturbation_stats: Dict,
    h1_summary: Dict,
    chosen_config: Dict,
    umap_results: pd.DataFrame,
) -> Path:
    """Write the 7-sheet Excel report.

    Parameters
    ----------
    df_analysis        : per-node statistics (output of analysis.analyze_nodes).
    sweep_df           : multiscale sweep results.
    bootstrap_df       : patient-level bootstrap probabilities.
    perturbation_stats : dict from stability.perturbation_stability.
    h1_summary         : dict from topological_exploration.analyze_h1_features.
    chosen_config      : dict with n_cubes, overlap.
    umap_results       : grid search results DataFrame.

    Returns
    -------
    output_path : Path to the written Excel file.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    output_path = OUTPUTS_DIR / "reporte_tda_final.xlsx"

    # Sheet 6 — hyperparameters
    hp_rows = [
        {"Parámetro": "seed", "Valor": 42},
        {"Parámetro": "n_cubes_elegido", "Valor": chosen_config["n_cubes"]},
        {"Parámetro": "overlap_elegido", "Valor": chosen_config["overlap"]},
        {"Parámetro": "dbscan_adaptive", "Valor": "True"},
        {"Parámetro": "bootstrap_n", "Valor": 50},
        {"Parámetro": "perturbation_n", "Valor": 20},
        {"Parámetro": "perturbation_sigma", "Valor": 0.01},
        {"Parámetro": "fdr_alpha", "Valor": 0.05},
        {"Parámetro": "cliff_delta_threshold", "Valor": 0.33},
        {"Parámetro": "pip_versions", "Valor": _pip_freeze_relevant()},
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
        df_analysis.to_excel(writer, sheet_name="nodos_estadisticas",    index=False)
        sweep_df.to_excel(   writer, sheet_name="barrido_multiescala",   index=False)
        bootstrap_df.to_excel(writer, sheet_name="estabilidad_bootstrap", index=False)
        df_pert.to_excel(    writer, sheet_name="estabilidad_perturbacion", index=False)
        df_h1.to_excel(      writer, sheet_name="homologia_persistente",  index=False)
        df_hp.to_excel(      writer, sheet_name="hiperparametros",        index=False)
        df_exec.to_excel(    writer, sheet_name="interpretacion_clinica", index=False)

    log.info("Excel report written: %s  (7 sheets)", output_path)
    return output_path
