"""Model selection: float-grouped 5-fold cross-validation over the 424 SWOT-Argo pairs (SI Text S3, Tables S4, S8).

    python scripts/01_select.py [--family ridge|pairs|pretrained]     # about an hour on a GPU for all three

The candidates are the input sets of Table S8 in three families: a ridge regression (1-6 modes, penalty 1, 10 or
100), the network trained on the pairs alone (5 members, 4 modes) and the pretrained network fine-tuned on the
pairs (8 members, 4 modes; also 40 levels for SSH + SST + MDT, and the inputs without SSH for Table S4). Each row
holds the CV RMSE, the RMSE of each fold and the R2 of the four mode coefficients. The pretrained SSH + SST + MDT
network is also cross-validated with DUACS input, for the SWOT-DUACS tie of Text S3 and Figure S5b.
The test sets are never loaded here. Rows already in results/cv.csv are skipped, so an interrupted run resumes.
Writes results/cv.csv and results/cv_predictions.npz (out-of-fold predictions of the pretrained rows); prints
the pick of the one-standard-error rule in each family.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from swot2argo import data, evaluate, models
from swot2argo.config import RESULTS_DIR

CV_FILE = os.path.join(RESULTS_DIR, "cv.csv")
OOF_FILE = os.path.join(RESULTS_DIR, "cv_predictions.npz")
INPUT_SETS = [("ssh",), ("ssh", "sst"), ("ssh", "mdt"), ("ssh", "sst", "mdt"), ("ssh", "sst", "mdt", "sss"),
              ("ssh", "sst", "mdt", "bathy"), ("ssh", "sst", "mdt", "latlon"), ("ssh", "sst", "mdt", "era5")]
NO_SSH = [("sst",), ("mdt",), ("sst", "mdt")]                       # Table S4 only
RIDGE_MODES, RIDGE_ALPHAS = (1, 2, 3, 4, 6), (1.0, 10.0, 100.0)
PAIRS_MEMBERS, PRETRAINED_MEMBERS = 5, 8
key = evaluate.candidate_key


class Table:
    def __init__(self):
        self.df = pd.read_csv(CV_FILE, dtype={"n_modes": str}) if os.path.exists(CV_FILE) else pd.DataFrame(columns=["key"])
        self.oof = dict(np.load(OOF_FILE)) if os.path.exists(OOF_FILE) else {}

    def has(self, k):
        return k in set(self.df["key"])

    def add(self, k, family, features, n_modes, alpha, ssh, cv, basis, pairs, t0):
        ct, cp = basis.encode(pairs.anom), basis.encode(cv["pred"] - pairs.clim)
        r2 = 1 - ((cp - ct) ** 2).sum(0) / ((ct - ct.mean(0)) ** 2).sum(0)
        row = dict(key=k, family=family, features=",".join(features), n_modes="levels" if n_modes is None else str(n_modes),
                   alpha=alpha, ssh=ssh, n_columns=models.n_columns(features), rmse=cv["rmse"],
                   **{f"fold_{i}": v for i, v in enumerate(cv["fold_rmse"])},
                   **{f"r2_mode{j + 1}": v for j, v in enumerate(r2)}, seconds=round(time.time() - t0))
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(CV_FILE, index=False)
        if family == "pretrained":
            self.oof[k] = cv["pred"].astype(np.float32)
            np.savez_compressed(OOF_FILE, **self.oof)
        print(f"{k:55s} CV {cv['rmse']:.4f}  ({row['seconds']} s)", flush=True)


def ridge(T, pairs, basis):
    for fs in INPUT_SETS:
        for k in RIDGE_MODES:
            for a in RIDGE_ALPHAS:
                if not T.has(key("ridge", fs, k, a)):
                    t0 = time.time()
                    cv = evaluate.cross_validate(lambda P: models.Ridge(fs, k, a).fit(P), pairs)
                    T.add(key("ridge", fs, k, a), "ridge", fs, k, a, "swot", cv, basis, pairs, t0)


def pairs_only(T, pairs, basis):
    for fs in INPUT_SETS:
        if not T.has(key("pairs", fs, 4)):
            t0 = time.time()
            cv = evaluate.cross_validate(lambda P: models.PairsNetwork(fs, 4, PAIRS_MEMBERS).fit(P), pairs)
            T.add(key("pairs", fs, 4), "pairs", fs, 4, np.nan, "swot", cv, basis, pairs, t0)


def pretrained(T, pairs, basis):
    runs = [(fs, 4, ("swot",)) for fs in INPUT_SETS + NO_SSH]
    runs[3] = (("ssh", "sst", "mdt"), 4, ("swot", "duacs"))
    runs.append((("ssh", "sst", "mdt"), None, ("swot",)))
    hist = data.history()
    for fs, k, sshs in runs:
        todo = [s for s in sshs if not T.has(key("pretrained", fs, k, ssh=s))]
        if not todo:
            continue
        t0 = time.time()
        base = models.Network(fs, k, PRETRAINED_MEMBERS).pretrain(hist)
        for s in todo:
            cv = evaluate.cross_validate(lambda P: base.finetune(P, ssh=s), pairs)
            T.add(key("pretrained", fs, k, ssh=s), "pretrained", fs, k, np.nan, s, cv, basis, pairs, t0)
            t0 = time.time()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["ridge", "pairs", "pretrained"], help="run one family only")
    a = ap.parse_args()
    P = data.pairs()
    basis = models.Target(4).fit(data.history().anom)          # the pretrained network's modes, for the per-mode R2
    print(f"{P.n} pairs from {len(np.unique(P.wmo))} floats, {evaluate.CV_FOLDS} float-grouped folds", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    T = Table()
    for fam, run in (("ridge", ridge), ("pairs", pairs_only), ("pretrained", pretrained)):
        if a.family in (None, fam):
            run(T, P, basis)
    for fam, (best, picked) in evaluate.selection(pd.read_csv(CV_FILE, dtype={"n_modes": str})).items():
        print(f"{fam:10s} picked {picked}   (lowest CV: {best})")
