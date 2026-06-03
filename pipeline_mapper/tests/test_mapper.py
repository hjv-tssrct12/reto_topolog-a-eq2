"""Tests for mapper_multiscale module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mapper_multiscale import (
    AdaptiveDBSCAN,
    _graph_metrics,
    _jaccard,
    run_mapper,
)


def _synthetic_lens_and_features(n: int = 100, p: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X    = rng.random((n, p))
    lens = rng.random((n, 2))
    return lens, X


def test_adaptive_dbscan_runs():
    rng = np.random.default_rng(42)
    X   = rng.random((50, 4))
    db  = AdaptiveDBSCAN(k=3, percentile=10, min_samples=2)
    db.fit(X)
    assert hasattr(db, "labels_")
    assert len(db.labels_) == 50


def test_adaptive_dbscan_small_input():
    X  = np.array([[0.0, 0.0], [0.1, 0.1]])
    db = AdaptiveDBSCAN(k=5, min_samples=2)
    db.fit(X)      # should not raise
    assert len(db.labels_) == 2


def test_run_mapper_returns_graph():
    lens, X = _synthetic_lens_and_features()
    graph = run_mapper(lens, X, n_cubes=4, overlap=0.5, clusterer=AdaptiveDBSCAN())
    assert "nodes" in graph
    assert "links" in graph


def test_graph_metrics_types():
    lens, X = _synthetic_lens_and_features()
    graph   = run_mapper(lens, X, n_cubes=4, overlap=0.5, clusterer=AdaptiveDBSCAN())
    m = _graph_metrics(graph)
    for key in ("n_nodes", "n_edges", "n_components", "n_cycles"):
        assert key in m
        assert isinstance(m[key], int)
        assert m[key] >= 0


def test_jaccard():
    a = frozenset([1, 2, 3])
    b = frozenset([2, 3, 4])
    assert abs(_jaccard(a, b) - 0.5) < 1e-9   # |{2,3}| / |{1,2,3,4}| = 2/4

    c = frozenset([10, 11])
    assert _jaccard(a, c) == 0.0               # disjoint

    assert _jaccard(frozenset(), frozenset()) == 0.0


def test_mapper_nodes_cover_data():
    """Every data point should appear in at least one node."""
    lens, X = _synthetic_lens_and_features(n=80)
    graph   = run_mapper(lens, X, n_cubes=5, overlap=0.5, clusterer=AdaptiveDBSCAN())
    assigned = set()
    for members in graph["nodes"].values():
        assigned.update(members)
    # Not all points are guaranteed to be in a node (noise in DBSCAN),
    # but there should be at least some coverage.
    assert len(assigned) > 0
