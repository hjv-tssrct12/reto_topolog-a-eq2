"""Phase 3: persistent homology of the 9-D feature space via Ripser."""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams
from ripser import ripser

from src.config import H1_PERCENTILE, OUTPUTS_DIR, RIPSER_MAX_DIM, RIPSER_THRESH

log = logging.getLogger(__name__)


def compute_persistence(X: np.ndarray) -> List[np.ndarray]:
    """Compute Vietoris-Rips persistence diagrams for H₀ and H₁.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Normalised feature matrix.

    Returns
    -------
    diagrams : list of np.ndarray
        diagrams[0] = H₀ pairs, diagrams[1] = H₁ pairs.
        Each array has shape (k, 2) with columns (birth, death).
    """
    log.info("Computing Vietoris-Rips persistence (maxdim=%d, thresh=%.2f) on %s ...",
             RIPSER_MAX_DIM, RIPSER_THRESH, X.shape)
    result = ripser(X, maxdim=RIPSER_MAX_DIM, thresh=RIPSER_THRESH)
    diagrams = result["dgms"]
    log.info("  H₀ features: %d | H₁ features: %d",
             len(diagrams[0]), len(diagrams[1]))
    return diagrams


def plot_persistence_diagram(diagrams: List[np.ndarray], output_path: Path) -> None:
    """Save persistence diagram as PNG.

    Parameters
    ----------
    diagrams : list of np.ndarray
        Output of compute_persistence.
    output_path : Path
        Destination file (PNG).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_diagrams(diagrams, ax=ax, show=False)
    ax.set_title("Persistence Diagram — H₀ y H₁ (Vietoris-Rips)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Persistence diagram saved: %s", output_path)


def compute_persistence_image(diagrams: List[np.ndarray]) -> np.ndarray:
    """Return a persistence landscape vector for H₁ features.

    Uses a simple binning approach compatible with persim 0.3.x.

    Parameters
    ----------
    diagrams : list of np.ndarray

    Returns
    -------
    pi_vector : np.ndarray, shape (n_bins,)
        Flattened persistence image (H₁ only).
    """
    h1 = diagrams[1]
    finite = h1[np.isfinite(h1[:, 1])]
    if len(finite) == 0:
        log.warning("No finite H₁ features — persistence image is zeros.")
        return np.zeros(100)

    persistence = finite[:, 1] - finite[:, 0]
    hist, _ = np.histogram(persistence, bins=100, range=(0, RIPSER_THRESH))
    return hist.astype(float)


def analyze_h1_features(diagrams: List[np.ndarray]) -> Dict:
    """Count H₁ features with persistence above the configured percentile.

    Parameters
    ----------
    diagrams : list of np.ndarray

    Returns
    -------
    summary : dict
        Keys: n_h0, n_h1, n_h1_persistent, threshold_used, has_circular_structure.
    """
    h1 = diagrams[1]
    finite = h1[np.isfinite(h1[:, 1])]
    persistence = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])

    if len(persistence) == 0:
        thresh_val = 0.0
        n_persistent = 0
    else:
        thresh_val = float(np.percentile(persistence, H1_PERCENTILE))
        n_persistent = int((persistence > thresh_val).sum())

    has_circular = n_persistent >= 1
    if has_circular:
        log.info("INFO: Se esperan estructuras circulares en el grafo Mapper "
                 "(%d ciclo(s) H₁ con persistencia > p%d=%.4f)",
                 n_persistent, H1_PERCENTILE, thresh_val)
    else:
        log.info("No se detectaron ciclos H₁ persistentes por encima del p%d.", H1_PERCENTILE)

    summary = {
        "n_h0"               : len(diagrams[0]),
        "n_h1"               : len(finite),
        "n_h1_persistent"    : n_persistent,
        "threshold_used"     : thresh_val,
        "has_circular_structure": has_circular,
    }
    return summary


def run_topological_exploration(X: np.ndarray) -> Dict:
    """Full Phase-3 pipeline: persistence, diagram PNG, PI vector, H₁ summary.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)

    Returns
    -------
    h1_summary : dict  (same as analyze_h1_features output)
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    diagrams = compute_persistence(X)

    plot_persistence_diagram(diagrams, OUTPUTS_DIR / "persistence_diagram.png")

    pi = compute_persistence_image(diagrams)
    np.save(OUTPUTS_DIR / "persistence_image.npy", pi)
    log.info("Persistence image vector saved (%d bins).", len(pi))

    h1_summary = analyze_h1_features(diagrams)
    return h1_summary
