"""Loading the data folder (the SWOT training pairs, the pre-2023 Argo record, the three test sets, the gridded
products at the test samples, the maps and the SWOT overpasses) and the results the scripts write.

Every set is a `Samples` object: a dictionary of equal-length arrays with attribute access.

    >>> from swot2argo import data
    >>> pairs = data.pairs()          # 424 Argo profiles within 24 h of a SWOT overpass
    >>> hist = data.history()         # 14,448 Argo profiles before 2023, with DUACS SSHA
    >>> tests = data.tests()          # {'moorings': ..., 'gliders': ..., 's2': ...}
    >>> pairs.dens.shape, pairs.anom.shape
    ((424, 40), (424, 40))
"""
import functools
import os

import numpy as np
import pandas as pd

from .config import DATA_DIR, DEPTH, RESULTS_DIR, TEST_SETS


class Samples(dict):
    """Arrays that share their first dimension. `s.lat`, `s["lat"]`, `s.n`, `s.dates`, `s.anom`, `s.subset(idx)`."""

    def __init__(self, name, arrays):
        super().__init__(arrays)
        self.name = name

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __repr__(self):
        cols = ", ".join(k for k in self if k not in ("dens", "clim"))
        return f"<Samples {self.name!r}: n={self.n}; dens/clim (n, {len(DEPTH)}); {cols}>"

    @property
    def n(self):
        return len(self["lat"])

    @property
    def dates(self):
        return pd.DatetimeIndex(self["time"])

    @property
    def anom(self):
        """Observed density minus the climatology: what the network predicts."""
        return self["dens"] - self["clim"]

    def subset(self, idx):
        return Samples(self.name, {k: v[np.asarray(idx)] for k, v in self.items()})


def _load(fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: put the data folder next to the code folder or set SWOT2ARGO_DATA")
    Z = np.load(path, allow_pickle=True)
    return {k: Z[k] for k in Z.files if k != "depth"}


@functools.lru_cache(maxsize=None)
def pairs():
    """The training set: 424 Argo profiles from 51 floats within 24 h and 3 km of a SWOT overpass (2023-2026)."""
    return Samples("pairs", _load("pairs.npz"))


@functools.lru_cache(maxsize=None)
def history():
    """The pretraining set: 14,448 Argo profiles from 145 floats before 2023 with DUACS SSHA, none from a float of
    the training pairs."""
    return Samples("history", _load("history.npz"))


@functools.lru_cache(maxsize=None)
def _tests_raw():
    return _load("tests.npz")


def test_set(name):
    R = _tests_raw()
    return Samples(name, {k[len(name) + 1:]: v for k, v in R.items() if k.startswith(name + "_")})


def tests():
    """{'moorings', 'gliders', 's2'} -> Samples. The zero-shot test sets: never used for any choice."""
    return {n: test_set(n) for n in TEST_SETS}


BENCHMARK_LABELS = {"glorys12": "GLORYS12 reanalysis", "glo12": "GLO12 analysis", "hycom": "HYCOM analysis",
                    "armor3d": "ARMOR3D multi-observation", "pwp": "PWP 1-D model"}


def benchmarks():
    """{product: {test: potential density (n, 40)}} at the test samples, aligned with `tests()`. GLORYS12, GLO12 and
    HYCOM assimilate altimetry and in situ profiles, ARMOR3D blends them statistically, PWP is a surface-forced
    column model. Samples a product does not cover are NaN."""
    Z = _load("benchmarks.npz")
    return {p: {nm: Z[f"{nm}_{p}"] for nm in TEST_SETS} for p in [str(x) for x in Z["products"]]}


@functools.lru_cache(maxsize=None)
def geo():
    """Station positions, SWOT swath edges and the coastline, for the maps."""
    return _load("geo.npz")


@functools.lru_cache(maxsize=None)
def s2_hourly():
    """The hourly S2 record, March 2023 - October 2024 (time, dens on the 40 levels): Figure 1c, persistence at S2
    (Table S3) and Figure S5a."""
    return _load("s2_hourly.npz")


@functools.lru_cache(maxsize=None)
def moorings_hourly():
    """The hourly record of each Cal/Val mooring, 20 March - 31 July 2023: {station: (time, dens (n, 40))}.
    Figure S5a only."""
    Z = _load("moorings_hourly.npz")
    return {str(st): (Z[f"{st}_time"], Z[f"{st}_dens"].astype(float)) for st in Z["stations"]}


@functools.lru_cache(maxsize=None)
def swath_crossovers():
    """SWOT minus DUACS SSHA where two SWOT passes less than 2 days apart cross (July 2023 - September 2026), one
    row per crossing: `var_cm2`, the variance of SWOT - DUACS on the crossing averaged over the two passes, and
    `cov_cm2`, its covariance between the two passes, both summed over scales of 2-240 km (cm2). Figure 3a."""
    return pd.read_csv(os.path.join(DATA_DIR, "swath_crossovers.csv"), parse_dates=["time"])


def swath(fname):
    """One SWOT overpass, data/swaths/<fname>: a dictionary of (n_lines, n_pixels) grids plus `time`, `pass_num`."""
    G = _load(os.path.join("swaths", fname))
    G["time"], G["pass_num"] = G["time"].astype("datetime64[s]")[()], int(G["pass_num"])
    return G


def swath_pixels(G):
    """The valid pixels of an overpass that carries the network's inputs, as a `Samples` (with `line`, `pixel` grid
    indices) that any model can predict."""
    li, pj = np.where(np.isfinite(G["ssh_swot"]) & np.isfinite(G["sst"]))
    return Samples("swath", dict(time=np.repeat(G["time"], len(li)), line=li, pixel=pj,
                                 clim=np.tile(G["clim"], (len(li), 1)),
                                 **{k: G[k][li, pj] for k in ("lat", "lon", "ssh_swot", "ssh_duacs", "sst", "mdt")}))


def station_latlon(station):
    G = geo()
    i = list(G["stations"]).index(station)
    return float(G["station_lat"][i]), float(G["station_lon"][i])


def bootstrap_blocks(S):
    """Resampling blocks for a test set: station x ISO week (moorings), day (gliders), ISO week (S2)."""
    d = S.dates
    iso = d.isocalendar()
    if S.name == "moorings":
        return pd.factorize(pd.Series(S["station"]) + "_" + pd.Series(iso.week.to_numpy()).astype(str))[0]
    if S.name == "gliders":
        return d.normalize().values.astype("datetime64[D]").astype(int)
    return (iso.year * 100 + iso.week).to_numpy(int)


# ----------------------------------------------------------------------------- results written by the scripts
PRED_FILE = os.path.join(RESULTS_DIR, "predictions.npz")


def save_predictions(P):
    """P: {'swot'|'duacs': {test: density (n, 40)}}"""
    np.savez_compressed(PRED_FILE, **{f"{s}|{t}": a for s, d in P.items() for t, a in d.items()})


def predictions():
    """{'swot'|'duacs': {test: potential density (n, 40)}} of the network, aligned with tests()."""
    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(f"{PRED_FILE} not found: run scripts/02_train.py (or keep the shipped results/)")
    Z = np.load(PRED_FILE)
    out = {}
    for k in Z.files:
        s, t = k.split("|")
        out.setdefault(s, {})[t] = Z[k]
    return out


def cv_predictions():
    """{candidate key: out-of-fold density (424, 40)} of the pretrained candidates of scripts/01_select.py."""
    return dict(np.load(os.path.join(RESULTS_DIR, "cv_predictions.npz")))


def scores():
    return pd.read_csv(os.path.join(RESULTS_DIR, "test_scores.csv"))


def gradients():
    return pd.read_csv(os.path.join(RESULTS_DIR, "gradients.csv"))
