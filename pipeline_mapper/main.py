"""Main orchestrator for the robust TDA pipeline.

Usage (from project root):
    python pipeline_tda/main.py
"""

import logging
import sys
import time
from pathlib import Path

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.config import OUTPUTS_DIR, SEED
from src.data_loading import load_and_clean, prepare_features
from src.topological_exploration import run_topological_exploration
from src.dim_reduction import umap_grid_search
from src.mapper_multiscale import (
    AdaptiveDBSCAN,
    find_persistent_nodes,
    multiscale_sweep,
    run_mapper,
)
from src.stability import bootstrap_stability, perturbation_stability
from src.analysis import analyze_nodes
from src.visualization import make_all_visualizations
from src.export import export_excel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(SEED)


def main() -> None:
    t0 = time.time()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    log.info("=" * 60)
    log.info("PIPELINE TDA MATERNO-NEONATAL  (seed=%d)", SEED)
    log.info("=" * 60)

    # ── Phase 1-2: Load & prepare ──────────────────────────────────────────
    log.info("[Phase 1-2] Loading and preparing data ...")
    df_work, lente_vals, cat_vals, cat_enc = load_and_clean()
    X_minmax, X_robust = prepare_features(df_work)

    # ── Phase 3: Topological exploration ───────────────────────────────────
    log.info("[Phase 3] Persistent homology ...")
    h1_summary = run_topological_exploration(X_minmax)

    # ── Phase 4: UMAP grid search ───────────────────────────────────────────
    log.info("[Phase 4] UMAP grid search (18 configs) ...")
    best_lens, umap_results, best_scaler = umap_grid_search(X_minmax, X_robust)
    X_features = X_minmax if best_scaler == "minmax" else X_robust
    log.info("  Best scaler: %s", best_scaler)

    # ── Phase 5: Mapper multiscale sweep ───────────────────────────────────
    log.info("[Phase 5] Mapper multiscale sweep (20 configs) ...")
    sweep_df, chosen_config, all_graphs, chosen_idx = multiscale_sweep(best_lens, X_features)
    log.info("  Chosen config: n_cubes=%d  overlap=%.2f",
             chosen_config["n_cubes"], chosen_config["overlap"])

    # Run final Mapper with chosen config
    final_graph = run_mapper(
        best_lens, X_features,
        chosen_config["n_cubes"], chosen_config["overlap"],
        AdaptiveDBSCAN(),
    )
    log.info("  Final graph: %d nodes  %d edges",
             len(final_graph["nodes"]),
             sum(len(v) for v in final_graph["links"].values()))

    # Persistent nodes across configs
    persistent_sets = find_persistent_nodes(all_graphs, chosen_idx)
    log.info("  Persistent node sets: %d", len(persistent_sets))

    # ── Phase 6: Stability ─────────────────────────────────────────────────
    log.info("[Phase 6a] Bootstrap stability (%d iterations) ...", 50)
    bootstrap_df = bootstrap_stability(
        X_features, best_lens, chosen_config, cat_vals,
    )

    log.info("[Phase 6b] Perturbation stability (%d iterations) ...", 20)
    perturbation_stats = perturbation_stability(
        X_features, best_lens, chosen_config, final_graph,
    )

    # ── Phase 8: Node analysis ─────────────────────────────────────────────
    log.info("[Phase 8] Statistical analysis per node ...")
    df_analysis = analyze_nodes(final_graph, lente_vals, cat_vals, df_work)

    # ── Phase 7: Visualizations ────────────────────────────────────────────
    log.info("[Phase 7] Generating visualizations ...")
    make_all_visualizations(final_graph, lente_vals, cat_enc, df_analysis)

    # ── Phase 10: Export ───────────────────────────────────────────────────
    log.info("[Phase 10] Exporting Excel report ...")
    report_path = export_excel(
        df_analysis     = df_analysis,
        sweep_df        = sweep_df,
        bootstrap_df    = bootstrap_df,
        perturbation_stats = perturbation_stats,
        h1_summary      = h1_summary,
        chosen_config   = chosen_config,
        umap_results    = umap_results,
    )

    # ── Final summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    n_sig   = int(df_analysis["nodo_significativo"].sum())

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETADO — {elapsed:.1f}s")
    print(f"  Nodos significativos      : {n_sig}")
    print(f"  Ciclos H₁ persistentes    : {h1_summary['n_h1_persistent']}")
    print(f"  Similitud perturbación    : {perturbation_stats['mean']:.3f} ± {perturbation_stats['std']:.3f}")
    print(f"  Config Mapper elegida     : n_cubes={chosen_config['n_cubes']}  overlap={chosen_config['overlap']}")
    print(f"  Excel report              : {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
