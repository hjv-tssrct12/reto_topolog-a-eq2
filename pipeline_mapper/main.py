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
from src.mapper_grid_search import mapper_grid_search          # ← Phase 5b
from src.stability import bootstrap_stability, perturbation_stability
from src.analysis import analyze_nodes
from src.visualization import make_all_visualizations
from src.export import export_excel
from src.ml_grid_search import (
    regression_grid_search,
    classification_grid_search,
)
from src.feature_analysis import feature_importance_analysis  # ← Phase 9b

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

    # ── Phase 5a: Mapper multiscale sweep ──────────────────────────────────
    log.info("[Phase 5a] Mapper multiscale sweep (%d configs) ...",
             len(__import__('src.config', fromlist=['MAPPER_N_CUBES_LIST']).MAPPER_N_CUBES_LIST)
             * len(__import__('src.config', fromlist=['MAPPER_OVERLAPS_LIST']).MAPPER_OVERLAPS_LIST))
    sweep_df, chosen_config, all_graphs, chosen_idx = multiscale_sweep(best_lens, X_features)
    log.info("  Chosen config (median nodes): n_cubes=%d  overlap=%.2f",
             chosen_config["n_cubes"], chosen_config["overlap"])

    # ── Phase 5b: Mapper grid search ───────────────────────────────────────
    log.info("[Phase 5b] Mapper grid search (multi-metric scoring) ...")
    gs_config, gs_results_df, gs_graph = mapper_grid_search(
        best_lens, X_features, cat_vals,
    )
    log.info(
        "  Best config (grid search): n_cubes=%d  overlap=%.2f  score=%.4f",
        gs_config["n_cubes"], gs_config["overlap"], gs_config["score"],
    )

    # ── Choose final config: prefer grid-search winner ─────────────────────
    # The grid search uses a richer multi-metric score (connectivity +
    # biological separation + cohesion + overlap richness + cycles),
    # so we adopt it as the final config.  The multiscale sweep result is
    # still exported for comparison.
    final_config = gs_config
    final_graph  = gs_graph
    log.info(
        "  Final config adopted: n_cubes=%d  overlap=%.2f",
        final_config["n_cubes"], final_config["overlap"],
    )
    log.info("  Final graph: %d nodes  %d edges",
             len(final_graph["nodes"]),
             sum(len(v) for v in final_graph["links"].values()))

    # Persistent nodes across multiscale configs
    persistent_sets = find_persistent_nodes(all_graphs, chosen_idx)
    log.info("  Persistent node sets: %d", len(persistent_sets))

    # ── Phase 6: Stability ─────────────────────────────────────────────────
    log.info("[Phase 6a] Bootstrap stability (%d iterations) ...", 50)
    bootstrap_df = bootstrap_stability(
        X_features, best_lens, final_config, cat_vals,
    )

    log.info("[Phase 6b] Perturbation stability (%d iterations) ...", 20)
    perturbation_stats = perturbation_stability(
        X_features, best_lens, final_config, final_graph,
    )

    # ── Phase 8: Node analysis ─────────────────────────────────────────────
    log.info("[Phase 8] Statistical analysis per node ...")
    df_analysis = analyze_nodes(final_graph, lente_vals, cat_vals, df_work)

    # ── Phase 9: Machine Learning ──────────────────────────────────────────
    log.info("[Phase 9] Machine Learning grid search ...")
    ml_reg_results = regression_grid_search(X_features, lente_vals)
    ml_clf_results = classification_grid_search(X_features, cat_vals)

    best_reg = ml_reg_results.iloc[0]
    best_clf = ml_clf_results.iloc[0]

    log.info("  Best regression model : %s (R²=%.4f)",
             best_reg["modelo"], best_reg["best_score"])
    log.info(
        "  Best classification model: %s  "
        "f1_macro=%.4f  acc=%.4f  "
        "[Bajo f1=%.3f | Medio f1=%.3f | Alto f1=%.3f]",
        best_clf["modelo"],
        best_clf["f1_macro"],
        best_clf["accuracy_cv"],
        best_clf.get("f1_Bajo",  0),
        best_clf.get("f1_Medio", 0),
        best_clf.get("f1_Alto",  0),
    )
    log.info("\n%s", ml_reg_results)
    log.info("\n%s", ml_clf_results[
        ["modelo", "f1_macro", "accuracy_cv",
         "f1_Bajo", "f1_Medio", "f1_Alto", "best_params"]
    ])

    # ── Phase 9b: Feature importance analysis ─────────────────────────────
    log.info("[Phase 9b] Feature importance analysis ...")
    from src.config import FEATURE_COLS
    df_feature_importance = feature_importance_analysis(
        X_features, lente_vals, cat_vals, list(FEATURE_COLS),
    )
    log.info("\n%s", df_feature_importance[
        ["feature", "eta_squared", "rf_imp_mean",
         "perm_imp_mean", "p_fdr", "kw_significativa", "composite_rank"]
    ].to_string(index=False))

    # ── Phase 7: Visualizations ────────────────────────────────────────────
    log.info("[Phase 7] Generating visualizations ...")
    make_all_visualizations(final_graph, lente_vals, cat_enc, df_analysis)

    # ── Phase 10: Export ───────────────────────────────────────────────────
    log.info("[Phase 10] Exporting Excel report ...")
    report_path = export_excel(
        df_analysis        = df_analysis,
        sweep_df           = sweep_df,
        bootstrap_df       = bootstrap_df,
        perturbation_stats = perturbation_stats,
        h1_summary         = h1_summary,
        chosen_config      = chosen_config,       # multiscale sweep winner
        umap_results       = umap_results,
        ml_reg_results     = ml_reg_results,
        ml_clf_results     = ml_clf_results,
        gs_results_df      = gs_results_df,       # ← grid search ranking
        gs_config          = gs_config,           # ← grid search winner
        df_feature_importance = df_feature_importance,  # ← Phase 9b
    )

    # ── Final summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    n_sig   = int(df_analysis["nodo_significativo"].sum())

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETADO — {elapsed:.1f}s")
    print(f"  Nodos significativos      : {n_sig}")
    print(f"  Ciclos H₁ persistentes    : {h1_summary['n_h1_persistent']}")
    print(f"  Similitud perturbación    : {perturbation_stats['mean']:.3f} ± {perturbation_stats['std']:.3f}")
    print(f"  Config sweep (mediana)    : n_cubes={chosen_config['n_cubes']}  overlap={chosen_config['overlap']}")
    print(f"  Config grid search        : n_cubes={gs_config['n_cubes']}  overlap={gs_config['overlap']}  score={gs_config['score']:.4f}")
    print(f"  Config final adoptada     : n_cubes={final_config['n_cubes']}  overlap={final_config['overlap']}")
    print(f"  Mejor regresor            : {best_reg['modelo']} (R²={best_reg['best_score']:.3f})")
    print(f"  Mejor clasificador        : {best_clf['modelo']} (f1_macro={best_clf['f1_macro']:.3f}  acc={best_clf['accuracy_cv']:.3f})")
    print(f"    Bajo  → f1={best_clf.get('f1_Bajo',  0):.3f}  precision={best_clf.get('precision_Bajo',  0):.3f}  recall={best_clf.get('recall_Bajo',  0):.3f}")
    print(f"    Medio → f1={best_clf.get('f1_Medio', 0):.3f}  precision={best_clf.get('precision_Medio', 0):.3f}  recall={best_clf.get('recall_Medio', 0):.3f}")
    print(f"    Alto  → f1={best_clf.get('f1_Alto',  0):.3f}  precision={best_clf.get('precision_Alto',  0):.3f}  recall={best_clf.get('recall_Alto',  0):.3f}")
    print(f"  Excel report              : {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()