"""Phase 6: bootstrap and perturbation stability analysis."""

import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    BOOTSTRAP_N,
    CAT_ORDER,
    OUTPUTS_DIR,
    PERTURBATION_N,
    PERTURBATION_SIGMA,
    SEED,
)
from src.mapper_multiscale import AdaptiveDBSCAN, Graph, _jaccard, _node_sets, run_mapper

log = logging.getLogger(__name__)


def _dominant_category(graph: Graph, cat_vals: np.ndarray) -> Dict[str, str]:
    """Return the dominant AF category for each node in a graph."""
    dom = {}
    for nid, members in graph["nodes"].items():
        counts = {c: int((cat_vals[members] == c).sum()) for c in CAT_ORDER}
        dom[nid] = max(counts, key=counts.get)
    return dom


def bootstrap_stability(
    X: np.ndarray,
    lens: np.ndarray,
    chosen_config: Dict,
    cat_vals: np.ndarray,
    n_iter: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> pd.DataFrame:
    """Estimate per-patient probability of landing in an Alto-AF dominant node.

    Parameters
    ----------
    X            : np.ndarray, shape (n, p)  normalised features.
    lens         : np.ndarray, shape (n, 2)  UMAP embedding.
    chosen_config: dict with keys n_cubes, overlap.
    cat_vals     : np.ndarray, shape (n,)    category labels.
    n_iter       : number of bootstrap iterations.
    seed         : random seed.

    Returns
    -------
    stability_df : pd.DataFrame with columns
        patient_id, prob_alto_af, prob_medio_af, prob_bajo_af, n_appearances.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    nc, ov = chosen_config["n_cubes"], chosen_config["overlap"]

    alto_counts  = np.zeros(n, dtype=int)
    medio_counts = np.zeros(n, dtype=int)
    bajo_counts  = np.zeros(n, dtype=int)
    appearances  = np.zeros(n, dtype=int)

    for i in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        X_b    = X[idx]
        lens_b = lens[idx]
        cats_b = cat_vals[idx]

        graph = run_mapper(lens_b, X_b, nc, ov, AdaptiveDBSCAN())
        dom   = _dominant_category(graph, cats_b)

        # Map bootstrap sample back to original indices
        patient_to_boot = {orig: [] for orig in range(n)}
        for boot_pos, orig_id in enumerate(idx):
            patient_to_boot[orig_id].append(boot_pos)

        for nid, members in graph["nodes"].items():
            cat = dom[nid]
            for boot_pos in members:
                orig_id = idx[boot_pos]
                appearances[orig_id] += 1
                if cat == "Alto":
                    alto_counts[orig_id] += 1
                elif cat == "Medio":
                    medio_counts[orig_id] += 1
                else:
                    bajo_counts[orig_id] += 1

        if (i + 1) % 10 == 0:
            log.info("Bootstrap: %d / %d done", i + 1, n_iter)

    safe = np.where(appearances > 0, appearances, 1)
    df = pd.DataFrame({
        "patient_id"  : np.arange(n),
        "prob_alto_af": np.round(alto_counts  / safe, 4),
        "prob_medio_af": np.round(medio_counts / safe, 4),
        "prob_bajo_af": np.round(bajo_counts  / safe, 4),
        "n_appearances": appearances,
    })
    df.to_csv(OUTPUTS_DIR / "stability_bootstrap.csv", index=False)
    log.info("Bootstrap stability saved: stability_bootstrap.csv")
    return df


def _mean_jaccard_between_graphs(g_ref: Graph, g_other: Graph) -> float:
    """Greedy mean Jaccard between nodes of two graphs."""
    ref_nodes   = list(_node_sets(g_ref).values())
    other_nodes = list(_node_sets(g_other).values())

    if not ref_nodes or not other_nodes:
        return 0.0

    total = 0.0
    for ref_set in ref_nodes:
        best = max((_jaccard(ref_set, o) for o in other_nodes), default=0.0)
        total += best
    return total / len(ref_nodes)


def perturbation_stability(
    X: np.ndarray,
    lens: np.ndarray,
    chosen_config: Dict,
    original_graph: Graph,
    n_iter: int = PERTURBATION_N,
    sigma: float = PERTURBATION_SIGMA,
    seed: int = SEED,
) -> Dict:
    """Assess graph stability under Gaussian noise on the feature matrix.

    Parameters
    ----------
    X             : np.ndarray, shape (n, p)
    lens          : np.ndarray, shape (n, 2)
    chosen_config : dict
    original_graph: the reference Mapper graph.
    n_iter        : perturbation iterations.
    sigma         : std-dev of additive Gaussian noise.
    seed          : random seed.

    Returns
    -------
    stats : dict with keys mean, std, all_similarities.
    """
    rng = np.random.default_rng(seed)
    nc, ov = chosen_config["n_cubes"], chosen_config["overlap"]
    similarities: List[float] = []

    for i in range(n_iter):
        noise = rng.normal(0, sigma, size=X.shape)
        X_pert = np.clip(X + noise, 0, 1)
        g_pert = run_mapper(lens, X_pert, nc, ov, AdaptiveDBSCAN())
        sim = _mean_jaccard_between_graphs(original_graph, g_pert)
        similarities.append(sim)

        if (i + 1) % 5 == 0:
            log.info("Perturbation: %d / %d  (sim=%.3f)", i + 1, n_iter, sim)

    stats = {
        "mean"            : round(float(np.mean(similarities)), 4),
        "std"             : round(float(np.std(similarities)),  4),
        "all_similarities": [round(s, 4) for s in similarities],
    }
    out = OUTPUTS_DIR / "stability_perturbation.json"
    out.write_text(json.dumps(stats, indent=2))
    log.info("Perturbation stability: mean=%.3f ± %.3f  saved: %s",
             stats["mean"], stats["std"], out)
    return stats
