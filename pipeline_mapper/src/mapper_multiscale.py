"""Phase 5: Mapper with multiple clusterers and multiscale parameter sweep."""

import logging
from typing import Dict, List, Tuple, Type

import kmapper as km
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

from src.config import (
    DBSCAN_KNN_K,
    DBSCAN_KNN_PERCENTILE,
    DBSCAN_MIN_SAMPLES,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    MAPPER_N_CUBES_LIST,
    MAPPER_OVERLAPS_LIST,
    OUTPUTS_DIR,
)

log = logging.getLogger(__name__)

Graph = Dict  # kmapper graph dict


class AdaptiveDBSCAN:
    """DBSCAN whose eps is set to a percentile of k-NN distances in the input cell.

    Parameters
    ----------
    k : int
        Neighbour rank used to measure local density.
    percentile : int
        Percentile of k-th-NN distances used as eps.
    min_samples : int
        Passed directly to DBSCAN.
    """

    def __init__(
        self,
        k: int = DBSCAN_KNN_K,
        percentile: int = DBSCAN_KNN_PERCENTILE,
        min_samples: int = DBSCAN_MIN_SAMPLES,
    ) -> None:
        self.k = k
        self.percentile = percentile
        self.min_samples = min_samples
        self.labels_: np.ndarray = np.array([])

    def get_params(self, deep: bool = True) -> dict:
        """sklearn-compatible interface required by kmapper."""
        return {"k": self.k, "percentile": self.percentile, "min_samples": self.min_samples}

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels (required by kmapper)."""
        self.fit(X)
        return self.labels_

    def fit(self, X: np.ndarray) -> "AdaptiveDBSCAN":
        """Fit DBSCAN with eps adapted to the local density of X."""
        if len(X) <= self.k:
            self.labels_ = np.zeros(len(X), dtype=int)
            return self

        nn = NearestNeighbors(n_neighbors=self.k + 1, algorithm="auto")
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        eps = float(np.percentile(distances[:, self.k], self.percentile))
        eps = max(eps, 1e-6)

        db = DBSCAN(eps=eps, min_samples=self.min_samples)
        db.fit(X)
        self.labels_ = db.labels_
        return self


class HDBSCANClusterer:
    """HDBSCAN wrapper compatible with kmapper's fit_predict interface.

    Falls back gracefully when cells have too few points.
    """

    def __init__(
        self,
        min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples: int = HDBSCAN_MIN_SAMPLES,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.labels_: np.ndarray = np.array([])

    def get_params(self, deep: bool = True) -> dict:
        return {"min_cluster_size": self.min_cluster_size, "min_samples": self.min_samples}

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def fit(self, X: np.ndarray) -> "HDBSCANClusterer":
        if len(X) <= self.min_cluster_size:
            self.labels_ = np.zeros(len(X), dtype=int)
            return self

        from sklearn.cluster import HDBSCAN
        hdb = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
        hdb.fit(X)
        labels = hdb.labels_.copy()
        if (labels == -1).all():
            labels = np.zeros(len(X), dtype=int)
        self.labels_ = labels
        return self


class AdaptiveAgglomerative:
    """AgglomerativeClustering (single linkage) with adaptive distance threshold.

    The threshold is derived from the same k-NN percentile used by
    AdaptiveDBSCAN, so both clusterers operate on a comparable density scale.
    """

    def __init__(
        self,
        k: int = DBSCAN_KNN_K,
        percentile: int = DBSCAN_KNN_PERCENTILE,
        linkage: str = "single",
    ) -> None:
        self.k = k
        self.percentile = percentile
        self.linkage = linkage
        self.labels_: np.ndarray = np.array([])

    def get_params(self, deep: bool = True) -> dict:
        return {"k": self.k, "percentile": self.percentile, "linkage": self.linkage}

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def fit(self, X: np.ndarray) -> "AdaptiveAgglomerative":
        if len(X) <= 1:
            self.labels_ = np.zeros(len(X), dtype=int)
            return self

        k = min(self.k, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        threshold = float(np.percentile(distances[:, k], self.percentile))
        threshold = max(threshold, 1e-6)

        agg = AgglomerativeClustering(
            n_clusters=None,
            linkage=self.linkage,
            distance_threshold=threshold,
        )
        agg.fit(X)
        self.labels_ = agg.labels_
        return self


# Ordered list of (name, class) pairs explored in every sweep.
# Add or remove entries here to change which clusterers are compared.
_CLUSTERERS: List[Tuple[str, Type]] = [
    ("AdaptiveDBSCAN",    AdaptiveDBSCAN),
    ("HDBSCAN",           HDBSCANClusterer),
    ("AdaptiveAgglom",    AdaptiveAgglomerative),
]


def build_clusterers() -> List[Tuple[str, Type]]:
    """Return the list of (name, class) pairs used in the parameter sweep."""
    return list(_CLUSTERERS)


def _graph_to_nx(graph: Graph) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(graph["nodes"].keys())
    for src, dests in graph["links"].items():
        for dst in dests:
            G.add_edge(src, dst)
    return G


def _graph_metrics(graph: Graph) -> Dict:
    """Compute topology metrics for a kmapper graph."""
    G = _graph_to_nx(graph)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_comp  = nx.number_connected_components(G)
    # first Betti number: independent cycles = E - V + C
    n_cycles = max(0, n_edges - n_nodes + n_comp)
    return {
        "n_nodes"     : n_nodes,
        "n_edges"     : n_edges,
        "n_components": n_comp,
        "n_cycles"    : n_cycles,
    }


def run_mapper(
    lens: np.ndarray,
    X: np.ndarray,
    n_cubes: int,
    overlap: float,
    clusterer: AdaptiveDBSCAN,
) -> Graph:
    """Run a single KeplerMapper instance.

    Parameters
    ----------
    lens : np.ndarray, shape (n, 1) or (n, 2)
    X    : np.ndarray, shape (n, p)
    n_cubes, overlap : Cover parameters.
    clusterer : AdaptiveDBSCAN instance.

    Returns
    -------
    graph : dict  (kmapper graph)
    """
    mapper = km.KeplerMapper(verbose=0)
    cover  = km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
    return mapper.map(lens, X, cover=cover, clusterer=clusterer)


def _node_sets(graph: Graph) -> Dict[str, frozenset]:
    return {nid: frozenset(members) for nid, members in graph["nodes"].items()}


def _jaccard(a: frozenset, b: frozenset) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def find_persistent_nodes(
    all_graphs: List[Graph],
    chosen_idx: int,
    jaccard_threshold: float = 0.7,
    min_config_frac: float   = 0.7,
) -> List[frozenset]:
    """Identify nodes in chosen_graph whose patient set recurs across configs.

    A node is "persistent" if a matching node (Jaccard ≥ jaccard_threshold)
    exists in at least min_config_frac of all other configs.

    Parameters
    ----------
    all_graphs      : list of all 20 kmapper graphs.
    chosen_idx      : index of the representative config in all_graphs.
    jaccard_threshold : minimum Jaccard for a match.
    min_config_frac : fraction of other configs that must contain a match.

    Returns
    -------
    persistent_sets : list of frozenset(patient_ids)
    """
    chosen_nodes = _node_sets(all_graphs[chosen_idx])
    other_graphs = [g for i, g in enumerate(all_graphs) if i != chosen_idx]
    if not other_graphs:
        return list(chosen_nodes.values())

    min_matches = int(np.ceil(min_config_frac * len(other_graphs)))
    persistent = []

    for node_id, node_set in chosen_nodes.items():
        match_count = 0
        for g in other_graphs:
            for other_set in _node_sets(g).values():
                if _jaccard(node_set, other_set) >= jaccard_threshold:
                    match_count += 1
                    break   # one match per config is enough
        if match_count >= min_matches:
            persistent.append(node_set)

    log.info("Persistent nodes: %d / %d (Jaccard≥%.1f in ≥%.0f%% of configs)",
             len(persistent), len(chosen_nodes),
             jaccard_threshold, min_config_frac * 100)
    return persistent


def multiscale_sweep(
    lens: np.ndarray,
    X: np.ndarray,
) -> Tuple[pd.DataFrame, Dict, List[Graph], int]:
    """Run Mapper for all (n_cubes × overlap × clusterer) combinations.

    Parameters
    ----------
    lens : np.ndarray, shape (n, 2)
    X    : np.ndarray, shape (n, p)

    Returns
    -------
    sweep_df     : pd.DataFrame with topology metrics for every config.
    chosen_config: dict with keys n_cubes, overlap, clusterer_name, clusterer.
    all_graphs   : flat list of Graph dicts (one per config, same order as sweep_df).
    chosen_idx   : index of chosen config in all_graphs / sweep_df.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    rows: List[Dict] = []
    all_graphs: List[Graph] = []

    clusterers = build_clusterers()
    clusterer_map: Dict[str, Type] = {name: cls for name, cls in clusterers}

    configs = [
        (nc, ov, cname, ccls)
        for cname, ccls in clusterers
        for nc in MAPPER_N_CUBES_LIST
        for ov in MAPPER_OVERLAPS_LIST
    ]
    log.info("Multiscale sweep: %d configs (%d clusterers × %d n_cubes × %d overlaps)",
             len(configs), len(clusterers), len(MAPPER_N_CUBES_LIST), len(MAPPER_OVERLAPS_LIST))

    for nc, ov, cname, ccls in configs:
        g = run_mapper(lens, X, nc, ov, ccls())
        metrics = _graph_metrics(g)
        all_graphs.append(g)
        rows.append({"n_cubes": nc, "overlap": ov, "clusterer": cname, **metrics})
        log.info("  n_cubes=%2d  overlap=%.2f  clusterer=%-18s  →  "
                 "nodes=%3d  edges=%4d  components=%d  cycles=%d",
                 nc, ov, cname,
                 metrics["n_nodes"], metrics["n_edges"],
                 metrics["n_components"], metrics["n_cycles"])

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(OUTPUTS_DIR / "mapper_multiscale.csv", index=False)
    log.info("Multiscale sweep saved: mapper_multiscale.csv  (%d rows)", len(sweep_df))

    # Choose the config whose n_nodes is closest to the global median
    median_nodes = sweep_df["n_nodes"].median()
    chosen_idx = int((sweep_df["n_nodes"] - median_nodes).abs().idxmin())
    chosen_row = sweep_df.iloc[chosen_idx]
    chosen_cname = chosen_row["clusterer"]
    chosen_config = {
        "n_cubes"       : int(chosen_row["n_cubes"]),
        "overlap"       : float(chosen_row["overlap"]),
        "clusterer_name": chosen_cname,
        "clusterer"     : clusterer_map[chosen_cname],
    }
    log.info(
        "Chosen config: n_cubes=%d  overlap=%.2f  clusterer=%s  (median n_nodes=%.1f)",
        chosen_config["n_cubes"], chosen_config["overlap"],
        chosen_config["clusterer_name"], median_nodes,
    )

    return sweep_df, chosen_config, all_graphs, chosen_idx
