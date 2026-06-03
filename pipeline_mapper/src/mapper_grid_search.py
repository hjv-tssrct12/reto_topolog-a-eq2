"""Phase 5b: Mapper grid search — exhaustive parameter search with multi-metric scoring.

Evaluates every (n_cubes × overlap) combination using a composite score that
balances topological richness, structural quality, and node-level cohesion.
A sigmoid density penalty discourages configurations that produce excessively
large graphs (too many small nodes → loss of statistical power in Phase 8).

Public API
----------
mapper_grid_search(lens, X, cat_vals) -> (best_config, results_df, best_graph)
"""

import logging
from itertools import product
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.config import (
    MAPPER_N_CUBES_LIST,
    MAPPER_OVERLAPS_LIST,
    OUTPUTS_DIR,
    SEED,
)
from src.mapper_multiscale import AdaptiveDBSCAN, Graph, _graph_to_nx, run_mapper

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Density / size penalty
# ---------------------------------------------------------------------------
# Configs with many small nodes reduce statistical power in Phase 8 (Mann-
# Whitney per node needs enough members to detect an effect).  We apply a
# smooth sigmoid penalty that is ≈1 for n_nodes ≤ TARGET and decays sharply
# above it, multiplying the composite score.
#
#   penalty(n) = 1 / (1 + exp(K * (n - TARGET)))
#
# Calibrated on the observed sweep range (97–366 nodes, N=1419 patients):
#   n ≤ 180  → penalty ≥ 0.87  (almost no discount)
#   n = 250  → penalty = 0.50  (half score)
#   n = 338  → penalty = 0.07  (near-zero)
_PENALTY_TARGET: int   = 250   # n_nodes at which score is halved
_PENALTY_K:      float = 0.03  # steepness of the sigmoid


def _size_penalty(n_nodes: int) -> float:
    """Sigmoid penalty in (0, 1]; decays for n_nodes > _PENALTY_TARGET."""
    return 1.0 / (1.0 + np.exp(_PENALTY_K * (n_nodes - _PENALTY_TARGET)))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _node_overlap_ratio(graph: Graph) -> float:
    """Fraction of point-node assignments that are shared (overlap signal).

    Returns 0 if no patient appears in more than one node (too crisp);
    higher values indicate richer overlapping structure — the core of Mapper.
    """
    all_assignments: List[int] = []
    for members in graph["nodes"].values():
        all_assignments.extend(members)
    if not all_assignments:
        return 0.0
    total         = len(all_assignments)
    unique_points = len(set(all_assignments))
    return (total - unique_points) / total


def _mean_node_size(graph: Graph) -> float:
    """Average number of patients per node."""
    sizes = [len(m) for m in graph["nodes"].values()]
    return float(np.mean(sizes)) if sizes else 0.0


def _node_size_cv(graph: Graph) -> float:
    """Coefficient of variation of node sizes (lower = more balanced)."""
    sizes = [len(m) for m in graph["nodes"].values()]
    if len(sizes) < 2:
        return 0.0
    mean = np.mean(sizes)
    return float(np.std(sizes) / mean) if mean > 0 else 0.0


def _graph_density(graph: Graph) -> float:
    """Edge density of the Mapper graph (edges / max_possible_edges)."""
    G = _graph_to_nx(graph)
    n = G.number_of_nodes()
    return nx.density(G) if n >= 2 else 0.0


def _lcc_fraction(graph: Graph) -> float:
    """Fraction of nodes in the largest connected component."""
    G = _graph_to_nx(graph)
    if G.number_of_nodes() == 0:
        return 0.0
    lcc_size = max(len(c) for c in nx.connected_components(G))
    return lcc_size / G.number_of_nodes()


def _mean_node_cohesion(graph: Graph, X: np.ndarray) -> float:
    """Average intra-node cohesion via silhouette score.

    Returns float in [-1, 1], higher is better.  Returns 0.0 if not
    computable (fewer than 2 nodes with ≥ 2 members).
    """
    node_items = [(nid, list(members)) for nid, members in graph["nodes"].items()
                  if len(members) >= 2]
    if len(node_items) < 2:
        return 0.0

    labels = np.full(len(X), -1, dtype=int)
    for node_idx, (_, members) in enumerate(node_items):
        for pt in members:
            if labels[pt] == -1:
                labels[pt] = node_idx

    mask = labels >= 0
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return 0.0

    try:
        return float(silhouette_score(
            X[mask], labels[mask], metric="euclidean",
            sample_size=min(500, int(mask.sum())),
            random_state=SEED,
        ))
    except Exception:
        return 0.0


def _category_separation(
    graph: Graph,
    cat_vals: np.ndarray,
    cat_order: Optional[List[str]] = None,
) -> float:
    """Normalised Gini impurity reduction across nodes vs. global mix.

    Higher → nodes are more category-pure (better AF-group separation).
    Returns float in [0, 1].
    """
    if cat_order is None:
        cat_order = list(np.unique(cat_vals))

    def _gini(members: List[int]) -> float:
        if not members:
            return 0.0
        counts = np.array(
            [(cat_vals[members] == c).sum() for c in cat_order], dtype=float
        )
        probs = counts / counts.sum()
        return float(1 - np.sum(probs ** 2))

    global_gini = _gini(list(range(len(cat_vals))))
    if global_gini == 0:
        return 0.0

    total_assigned = sum(len(m) for m in graph["nodes"].values())
    if total_assigned == 0:
        return 0.0

    weighted_gini = sum(
        len(members) * _gini(list(members))
        for members in graph["nodes"].values()
    ) / total_assigned

    return float(max(0.0, (global_gini - weighted_gini) / global_gini))


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalise_column(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; constant columns → 0."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------
# Weights must sum to 1.0.  Tune here to shift the search objective.
_METRIC_WEIGHTS: Dict[str, float] = {
    "lcc_fraction"   : 0.25,   # connectivity — prefer one coherent structure
    "cat_separation" : 0.25,   # biological relevance — AF groups separated
    "node_cohesion"  : 0.20,   # intra-node compactness
    "overlap_ratio"  : 0.15,   # Mapper-specific overlap richness
    "n_cycles_norm"  : 0.15,   # topological loops (β₁)
}


# ---------------------------------------------------------------------------
# Main grid search
# ---------------------------------------------------------------------------

def mapper_grid_search(
    lens: np.ndarray,
    X: np.ndarray,
    cat_vals: np.ndarray,
    n_cubes_list:  Optional[List[int]]   = None,
    overlaps_list: Optional[List[float]] = None,
    cat_order:     Optional[List[str]]   = None,
) -> Tuple[Dict, pd.DataFrame, Graph]:
    """Exhaustive grid search over Mapper (n_cubes, overlap) pairs.

    For each configuration the function:
      1. Runs KeplerMapper with AdaptiveDBSCAN.
      2. Computes five quality metrics.
      3. Normalises metrics across configs and computes a weighted composite.
      4. Applies a sigmoid size penalty to discount overly-large graphs.
      5. Selects the configuration with the highest penalised score.

    Size penalty
    ------------
    score_final = composite_score × penalty(n_nodes)

    penalty(n) = 1 / (1 + exp(K × (n − TARGET)))

    With TARGET={target} and K={k}, configs with ≤180 nodes are almost
    unaffected (penalty ≥ 0.87) while configs above 280 nodes are heavily
    discounted (penalty < 0.20), preventing the selection of graphs too
    fragmented for per-node statistical testing.

    Parameters
    ----------
    lens          : np.ndarray, shape (n, 2)  — UMAP lens from Phase 4.
    X             : np.ndarray, shape (n, p)  — scaled feature matrix.
    cat_vals      : np.ndarray, shape (n,)    — AF category labels (str).
    n_cubes_list  : override MAPPER_N_CUBES_LIST from config.
    overlaps_list : override MAPPER_OVERLAPS_LIST from config.
    cat_order     : ordinal category list (inferred from cat_vals if None).

    Returns
    -------
    best_config : dict — {{"n_cubes": int, "overlap": float,
                           "composite_score": float, "penalised_score": float,
                           "size_penalty": float}}
    results_df  : pd.DataFrame — all configs ranked by penalised_score (desc).
    best_graph  : Graph — kmapper graph for the best config.
    """.format(target=_PENALTY_TARGET, k=_PENALTY_K)

    OUTPUTS_DIR.mkdir(exist_ok=True)

    if n_cubes_list  is None: n_cubes_list  = MAPPER_N_CUBES_LIST
    if overlaps_list is None: overlaps_list = MAPPER_OVERLAPS_LIST

    configs = list(product(n_cubes_list, overlaps_list))
    total   = len(configs)
    log.info("Mapper grid search: %d configurations  (penalty target=%d nodes)",
             total, _PENALTY_TARGET)

    rows:   List[Dict]  = []
    graphs: List[Graph] = []

    for i, (nc, ov) in enumerate(configs, start=1):
        log.info("[Mapper GS %d/%d] n_cubes=%d  overlap=%.2f", i, total, nc, ov)

        g = run_mapper(lens, X, nc, ov, AdaptiveDBSCAN())
        graphs.append(g)

        G        = _graph_to_nx(g)
        n_nodes  = G.number_of_nodes()
        n_edges  = G.number_of_edges()
        n_comp   = nx.number_connected_components(G) if n_nodes > 0 else 0
        n_cycles = max(0, n_edges - n_nodes + n_comp)

        rows.append({
            "n_cubes"       : nc,
            "overlap"       : ov,
            "n_nodes"       : n_nodes,
            "n_edges"       : n_edges,
            "n_components"  : n_comp,
            "n_cycles"      : n_cycles,
            "overlap_ratio" : round(_node_overlap_ratio(g),          4),
            "mean_node_size": round(_mean_node_size(g),               2),
            "size_cv"       : round(_node_size_cv(g),                 4),
            "graph_density" : round(_graph_density(g),                4),
            "lcc_fraction"  : round(_lcc_fraction(g),                 4),
            "node_cohesion" : round(_mean_node_cohesion(g, X),        4),
            "cat_separation": round(_category_separation(g, cat_vals, cat_order), 4),
            "size_penalty"  : round(_size_penalty(n_nodes),           4),
        })

        log.info(
            "  → nodes=%3d  edges=%4d  comp=%2d  cycles=%3d  "
            "lcc=%.2f  cat_sep=%.3f  cohesion=%.3f  penalty=%.3f",
            n_nodes, n_edges, n_comp, n_cycles,
            rows[-1]["lcc_fraction"], rows[-1]["cat_separation"],
            rows[-1]["node_cohesion"], rows[-1]["size_penalty"],
        )

    # ── Normalise metrics & compute composite ─────────────────────────────
    norm = pd.DataFrame(rows)
    norm["lcc_fraction_n"]   = _normalise_column(norm["lcc_fraction"])
    norm["cat_separation_n"] = _normalise_column(norm["cat_separation"])
    norm["node_cohesion_n"]  = _normalise_column(norm["node_cohesion"])
    norm["overlap_ratio_n"]  = _normalise_column(norm["overlap_ratio"])
    norm["n_cycles_norm"]    = _normalise_column(norm["n_cycles"].astype(float))

    composite = (
        _METRIC_WEIGHTS["lcc_fraction"]   * norm["lcc_fraction_n"]
      + _METRIC_WEIGHTS["cat_separation"] * norm["cat_separation_n"]
      + _METRIC_WEIGHTS["node_cohesion"]  * norm["node_cohesion_n"]
      + _METRIC_WEIGHTS["overlap_ratio"]  * norm["overlap_ratio_n"]
      + _METRIC_WEIGHTS["n_cycles_norm"]  * norm["n_cycles_norm"]
    ).round(4)

    # ── Apply size penalty ─────────────────────────────────────────────────
    penalised = (composite * norm["size_penalty"]).round(4)

    results_df = pd.DataFrame(rows)
    results_df["composite_score"] = composite
    results_df["penalised_score"] = penalised
    results_df.sort_values("penalised_score", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    results_df.to_csv(OUTPUTS_DIR / "mapper_grid_search.csv", index=False)
    log.info("Mapper grid search saved: mapper_grid_search.csv")

    # ── Best config ────────────────────────────────────────────────────────
    best_row = results_df.iloc[0]
    best_config = {
        "n_cubes"         : int(best_row["n_cubes"]),
        "overlap"         : float(best_row["overlap"]),
        "score"           : float(best_row["penalised_score"]),   # kept as "score" for compatibility
        "composite_score" : float(best_row["composite_score"]),
        "penalised_score" : float(best_row["penalised_score"]),
        "size_penalty"    : float(best_row["size_penalty"]),
    }

    # Rebuild graph deterministically for the best config
    best_graph = run_mapper(
        lens, X,
        best_config["n_cubes"],
        best_config["overlap"],
        AdaptiveDBSCAN(),
    )

    log.info(
        "Best Mapper config: n_cubes=%d  overlap=%.2f  "
        "composite=%.4f  penalty=%.3f  penalised=%.4f",
        best_config["n_cubes"], best_config["overlap"],
        best_config["composite_score"], best_config["size_penalty"],
        best_config["penalised_score"],
    )
    log.info(
        "  Metrics — lcc=%.3f  cat_sep=%.3f  cohesion=%.3f  "
        "overlap=%.3f  cycles=%d  n_nodes=%d",
        best_row["lcc_fraction"], best_row["cat_separation"],
        best_row["node_cohesion"], best_row["overlap_ratio"],
        best_row["n_cycles"], best_row["n_nodes"],
    )

    return best_config, results_df, best_graph