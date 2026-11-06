"""Paths and the fixed choices of the study."""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))               # the repository


def _data_dir():
    """SWOT2ARGO_DATA if set; else data/ inside the repository; else a data/ folder next to it."""
    if "SWOT2ARGO_DATA" in os.environ:
        return os.environ["SWOT2ARGO_DATA"]
    inside = os.path.join(ROOT, "data")
    return inside if os.path.isdir(inside) else os.path.join(os.path.dirname(ROOT), "data")


DATA_DIR = _data_dir()
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

DEPTH = np.arange(40) * 5.0            # the 40 levels, 0-195 m every 5 m

# The network: inputs, target and ensemble size (Section 4 of the paper)
FEATURES = ("ssh", "sst", "mdt")       # SSHA (SWOT or DUACS), OSTIA SST, MDT
N_MODES = 4                            # the leading PCA modes of the density anomaly
N_MEMBERS = 15                         # networks per ensemble; the prediction is their mean

# Selection (SI Text S3): float-grouped cross-validation over the pairs; inputs a final model may not use
CV_FOLDS = 5
EXCLUDED_INPUTS = ("bathy", "latlon", "era5")      # gains lost out of the sampled region, or mostly the time of year

# Block bootstrap for the confidence intervals: blocks are station x ISO week (moorings), day (gliders), ISO week (S2)
N_BOOT = 1000

TEST_SETS = ("moorings", "gliders", "s2")
