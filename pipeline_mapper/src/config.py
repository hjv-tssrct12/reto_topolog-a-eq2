"""Centralised configuration: paths, column names, and all hyperparameters."""

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parents[1]   # pipeline_tda/
PROJECT_ROOT  = _PIPELINE_DIR.parent                  # repo root

BD_PATH     = PROJECT_ROOT / "BD" / "DB_Limpia_TDA.xlsx"
OUTPUTS_DIR = _PIPELINE_DIR / "outputs"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
LENTE_COL = "Total mg/d AF  suple y pan"
CAT_COL   = "Total mg/d AF  suple y pan [cat]"
CAT_ORDER = ["Bajo", "Medio", "Alto"]

MATERNAS   = ["Edad madre", "Educación", "N° embarazo", "Región", "Dif peso mamá", "Dif IMC"]
NEONATALES = ["Sexo hijo", "PN hijo (g)", "EG hijo (sem)"]
FEATURE_COLS = MATERNAS + NEONATALES

CAT_COLORS = {"Bajo": "#4C72B0", "Medio": "#DD8452", "Alto": "#55A868"}

# ---------------------------------------------------------------------------
# UMAP grid search — n_neighbors based on sqrt(N), N=1419
# ---------------------------------------------------------------------------
_N = 1419
_SQ = int(math.sqrt(_N))                        # 37
UMAP_N_NEIGHBORS_CANDIDATES = [max(2, _SQ // 2), _SQ, _SQ * 2]   # [18, 37, 74]
UMAP_MIN_DIST_CANDIDATES    = [0.0, 0.1, 0.3]
UMAP_TRUST_N_NEIGHBORS      = 5                 # used for trustworthiness / continuity

# ---------------------------------------------------------------------------
# Mapper multiscale sweep
# ---------------------------------------------------------------------------
MAPPER_N_CUBES_LIST  = [6, 8, 10, 12, 15]
MAPPER_OVERLAPS_LIST = [0.25, 0.4, 0.5, 0.6]

# ---------------------------------------------------------------------------
# Adaptive DBSCAN (per-cell)
# ---------------------------------------------------------------------------
DBSCAN_KNN_K          = 5
DBSCAN_KNN_PERCENTILE = 10
DBSCAN_MIN_SAMPLES    = 3

# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
BOOTSTRAP_N        = 50
PERTURBATION_N     = 20
PERTURBATION_SIGMA = 0.01

# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------
FDR_ALPHA             = 0.05
CLIFF_DELTA_THRESHOLD = 0.33
CI_BOOTSTRAP_N        = 1000

# ---------------------------------------------------------------------------
# Persistence homology
# ---------------------------------------------------------------------------
RIPSER_MAX_DIM = 1          # compute H₀ and H₁
RIPSER_THRESH  = 0.8        # limits filtration scale for speed
H1_PERCENTILE  = 90         # threshold for "persistent" H₁ feature
