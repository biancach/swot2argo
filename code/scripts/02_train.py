"""Train the networks of the paper on all 424 SWOT-Argo pairs and predict the test sets.

    python scripts/02_train.py [--members 15] [--only headline|table-s8]    # minutes for the headline, ~30 min in all

1. The reported model, the pick of scripts/01_select.py: stage 1 pretrains the ensemble on the 14,448 pre-2023 Argo
   profiles with DUACS SSHA, SST and MDT; stage 2 fine-tunes every layer on the pairs, once with SWOT input and once
   with DUACS input (the comparison of the paper). Writes results/models/headline_swot.pt, headline_duacs.pt and
   results/predictions.npz.
2. The other configurations of SI Table S8, with SWOT input: results/configurations.npz.
--members 2 makes a quick smoke test. Both outputs replace the shipped ones.
"""
import argparse
import os
import time

import numpy as np

from swot2argo import data, models
from swot2argo.config import FEATURES, MODELS_DIR, N_MEMBERS, RESULTS_DIR
from swot2argo.evaluate import candidate_key

CONFIG_FILE = os.path.join(RESULTS_DIR, "configurations.npz")
# the rows of Table S8 besides the reported model: (family, inputs, modes or None for 40 levels, ridge penalty)
TABLE_S8 = [("ridge", ("ssh",), 1, 1.0), ("ridge", FEATURES, 4, 10.0),
            ("pairs", FEATURES, 4, np.nan), ("pairs", FEATURES + ("sss",), 4, np.nan),
            ("pretrained", ("ssh",), 4, np.nan), ("pretrained", ("ssh", "sst"), 4, np.nan),
            ("pretrained", ("ssh", "mdt"), 4, np.nan), ("pretrained", FEATURES + ("sss",), 4, np.nan),
            ("pretrained", FEATURES + ("bathy",), 4, np.nan), ("pretrained", FEATURES + ("latlon",), 4, np.nan),
            ("pretrained", FEATURES + ("era5",), 4, np.nan), ("pretrained", FEATURES, None, np.nan)]


def headline(pairs, hist, tests, n_members):
    t0 = time.time()
    base = models.Network(n_members=n_members).pretrain(hist)
    print(f"{base}  ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    preds = {}
    for ssh in ("swot", "duacs"):
        m = base.finetune(pairs, ssh=ssh)
        models.save(m, os.path.join(MODELS_DIR, f"headline_{ssh}.pt"))
        preds[ssh] = {nm: m.predict(S) for nm, S in tests.items()}
        print(m, flush=True)
    data.save_predictions(preds)
    print("wrote", data.PRED_FILE)


def configurations(pairs, hist, tests, n_members):
    out = {}
    for family, fs, k, alpha in TABLE_S8:
        t0 = time.time()
        if family == "ridge":
            m = models.Ridge(fs, k, alpha).fit(pairs)
        elif family == "pairs":
            m = models.PairsNetwork(fs, k, n_members).fit(pairs)
        else:
            m = models.Network(fs, k, n_members).pretrain(hist, verbose=False).finetune(pairs, ssh="swot")
        name = candidate_key(family, fs, k, alpha)
        out.update({f"{name}#{nm}": m.predict(S).astype(np.float32) for nm, S in tests.items()})
        print(f"{name:45s} ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(CONFIG_FILE, **out)
    print("wrote", CONFIG_FILE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", type=int, default=N_MEMBERS, help=f"networks per ensemble (default {N_MEMBERS})")
    ap.add_argument("--only", choices=["headline", "table-s8"], help="train only the reported model or only the rest")
    a = ap.parse_args()
    pairs, hist, tests = data.pairs(), data.history(), data.tests()
    assert not set(pairs.wmo) & set(hist.wmo), "a training-pair float is in the pretraining record"
    if a.only in (None, "headline"):
        headline(pairs, hist, tests, a.members)
    if a.only in (None, "table-s8"):
        configurations(pairs, hist, tests, a.members)
