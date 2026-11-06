"""Scores. Cross-validation over the SWOT-Argo pairs, which alone chooses the model (SI Text S3), and the scores at
the zero-shot test sets: RMSE with block-bootstrap confidence intervals, the SWOT-versus-DUACS difference, and the
same difference binned by how far the two inputs disagree at the sample."""
import numpy as np
import pandas as pd

from . import data
from .config import CV_FOLDS, DEPTH, EXCLUDED_INPUTS, N_BOOT, TEST_SETS
from .physics import mld

BANDS = {"0-30 m": DEPTH <= 30, "30-100 m": (DEPTH > 30) & (DEPTH <= 100), "100-200 m": DEPTH > 100}


def rmse(pred, truth, levels=slice(None)):
    """RMSE over samples and levels (kg/m3)."""
    return float(np.sqrt(np.nanmean((np.asarray(pred)[:, levels] - np.asarray(truth)[:, levels]) ** 2)))


# ----------------------------------------------------------------------------- cross-validation (selection)
def folds(pairs, n_folds=CV_FOLDS, seed=0):
    """Fold of each pair: whole floats per fold, the same assignment for every candidate."""
    floats = np.random.default_rng(seed).permutation(np.unique(pairs.wmo))
    fold_of = {f: i % n_folds for i, f in enumerate(floats)}
    return np.array([fold_of[w] for w in pairs.wmo])


def cross_validate(fit, pairs, n_folds=CV_FOLDS):
    """fit(training pairs) -> a fitted model. Returns the out-of-fold density predictions of every pair, their RMSE
    and the RMSE of each fold."""
    fold = folds(pairs, n_folds)
    pred = np.full_like(pairs.dens, np.nan)
    for f in range(n_folds):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        assert not set(pairs.wmo[tr]) & set(pairs.wmo[te])
        pred[te] = fit(pairs.subset(tr)).predict(pairs.subset(te))
    fold_rmse = [rmse(pred[fold == f], pairs.dens[fold == f]) for f in range(n_folds)]
    return dict(pred=pred, rmse=rmse(pred, pairs.dens), fold_rmse=fold_rmse)


def candidate_key(family, features, n_modes, alpha=np.nan, ssh="swot"):
    """'family|inputs|modes|penalty|ssh', the name of a candidate in results/cv.csv and results/configurations.npz."""
    a = "" if np.isnan(alpha) else f"{alpha:g}"
    return f"{family}|{','.join(features)}|{'levels' if n_modes is None else n_modes}|{a}|{ssh}"


def pick(table, simplicity):
    """The one-standard-error rule on a table of candidates (columns rmse and fold_0, fold_1, ...): among the rows
    whose mean per-fold difference from the best row is within one paired standard error, the simplest, by the
    columns `simplicity` (ascending). Returns (index of the best row, index of the picked row)."""
    fcols = [c for c in table.columns if c.startswith("fold_")]
    best = table["rmse"].idxmin()
    diff = table[fcols].to_numpy() - table.loc[best, fcols].to_numpy(float)
    se = diff.std(1, ddof=1) / np.sqrt(len(fcols))
    ok = table.index[(diff.mean(1) <= se) | (table.index == best)]
    return best, table.loc[ok].sort_values(list(simplicity) + ["rmse"]).index[0]


def selection(cv):
    """{family: (key of the lowest-CV row, key of the picked row)} for a table of candidates like results/cv.csv:
    the one-standard-error rule among the rows a final model may come from (4 modes, SWOT input, none of the
    excluded inputs); simplest = fewest input columns, then the strongest ridge penalty."""
    out = {}
    for fam, t in cv.groupby("family"):
        ok = (t.n_modes.astype(str) == "4") & (t.ssh == "swot") & ~t.features.str.contains("|".join(EXCLUDED_INPUTS))
        t = t[ok].assign(neg_alpha=-t.alpha.fillna(0))
        best, picked = pick(t, ["n_columns", "neg_alpha"])
        out[fam] = (t.loc[best, "key"], t.loc[picked, "key"])
    return out


# ----------------------------------------------------------------------------- the test sets


def boot_picks(blocks, n_boot=N_BOOT, seed=0):
    """Index arrays of n_boot block-bootstrap resamples."""
    u = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in u}
    rng = np.random.default_rng(seed)
    return [np.concatenate([idx[b] for b in rng.choice(u, len(u), replace=True)]) for _ in range(n_boot)]


def bootstrap_rmse(pred, S, n_boot=N_BOOT):
    """90% block-bootstrap interval of RMSE(pred) at test set S -> (lo, hi). NaN rows (uncovered samples) are dropped."""
    ok = np.isfinite(pred).all(1)
    S, pred = S.subset(ok), pred[ok]
    e = ((pred - S.dens) ** 2).mean(1)
    r = [np.sqrt(e[p].mean()) for p in boot_picks(data.bootstrap_blocks(S), n_boot)]
    lo, hi = np.percentile(r, [5, 95])
    return float(lo), float(hi)


def bootstrap_vs(pred, other, S, n_boot=N_BOOT):
    """RMSE(pred) - RMSE(other) at S with its 90% interval; negative = pred better. dict(diff, lo, hi, significant)."""
    ok = np.isfinite(pred).all(1) & np.isfinite(other).all(1)
    S, pred, other = S.subset(ok), pred[ok], other[ok]
    e1, e0 = ((pred - S.dens) ** 2).mean(1), ((other - S.dens) ** 2).mean(1)
    d = np.array([np.sqrt(e1[p].mean()) - np.sqrt(e0[p].mean()) for p in boot_picks(data.bootstrap_blocks(S), n_boot)])
    lo, hi = np.percentile(d, [5, 95])
    return dict(diff=float(np.sqrt(e1.mean()) - np.sqrt(e0.mean())), lo=float(lo), hi=float(hi),
                significant=bool(lo > 0 or hi < 0))


def score_table(estimates, tests):
    """estimates: {label: {test: density (n, 40)}} -> one row per label: RMSE 0-200 m with its 90% interval, RMSE
    by depth band, and the difference from the climatology (d_*, negative = better) with its interval."""
    rows = []
    for lab, P in estimates.items():
        r = dict(model=lab)
        for nm in TEST_SETS:
            if nm not in P:
                continue
            S = tests[nm]
            r[f"n_{nm}"] = int(np.isfinite(P[nm]).all(1).sum())
            r[nm] = rmse(P[nm], S.dens)
            r[f"{nm}_lo"], r[f"{nm}_hi"] = bootstrap_rmse(P[nm], S)
            for bl, levels in BANDS.items():
                r[f"{nm} {bl}"] = rmse(P[nm], S.dens, levels)
            b = bootstrap_vs(P[nm], S.clim, S)
            r[f"d_{nm}"], r[f"d_{nm}_lo"], r[f"d_{nm}_hi"], r[f"d_{nm}_sig"] = b["diff"], b["lo"], b["hi"], b["significant"]
        rows.append(r)
    return pd.DataFrame(rows)


def swot_vs_duacs(preds, tests):
    """preds: {'swot': {test: density}, 'duacs': {...}} -> one row per test set: RMSE fed SWOT minus fed DUACS."""
    rows = []
    for nm in TEST_SETS:
        b = bootstrap_vs(preds["swot"][nm], preds["duacs"][nm], tests[nm])
        rows.append(dict(test=nm, swot=rmse(preds["swot"][nm], tests[nm].dens),
                         duacs=rmse(preds["duacs"][nm], tests[nm].dens), **b))
    return pd.DataFrame(rows)


def disagreement_bins(preds, tests, n_boot=N_BOOT):
    """The same difference with the samples binned by |SWOT - DUACS SSHA| at the sample: below its median, median to
    75th percentile, 75th to 90th, and the top 10%. One row per test set and bin."""
    rows = []
    for nm in TEST_SETS:
        S = tests[nm]
        dssh = np.abs(S.ssh_swot - S.ssh_duacs)
        es = ((preds["swot"][nm] - S.dens) ** 2).mean(1)
        ed = ((preds["duacs"][nm] - S.dens) ** 2).mean(1)
        q = np.percentile(dssh, [50, 75, 90])
        picks = boot_picks(data.bootstrap_blocks(S), n_boot)
        for lo, hi, lab in [(0, q[0], "< p50"), (q[0], q[1], "p50-p75"), (q[1], q[2], "p75-p90"), (q[2], np.inf, "> p90")]:
            m = (dssh >= lo) & (dssh < hi)
            d = np.array([np.sqrt(es[p][m[p]].mean()) - np.sqrt(ed[p][m[p]].mean()) for p in picks if m[p].sum() >= 5])
            clo, chi = np.percentile(d, [5, 95])
            rows.append(dict(test=nm, bin=lab, dssh_cm_from=100 * lo, n=int(m.sum()), swot=float(np.sqrt(es[m].mean())),
                             duacs=float(np.sqrt(ed[m].mean())), diff=float(np.sqrt(es[m].mean()) - np.sqrt(ed[m].mean())),
                             lo=float(clo), hi=float(chi), significant=bool(clo > 0 or chi < 0)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- other baselines and diagnostics
def persistence(S, lag_days, hourly=None):
    """The profile observed at the same site `lag_days` earlier: at the moorings the same station's sample nearest
    to t - lag within 12 h; at S2 the `hourly` S2 record (time, dens) within 3 h. NaN rows where there is none."""
    def nearest(t, t_src, tol_h):
        want = t - np.timedelta64(int(lag_days * 24), "h")
        j = np.clip(np.searchsorted(t_src, want), 1, len(t_src) - 1)
        j = np.where(np.abs(t_src[j - 1] - want) <= np.abs(t_src[j] - want), j - 1, j)
        return np.where(np.abs(t_src[j] - want) <= np.timedelta64(tol_h, "h"), j, -1)

    out = np.full_like(S.dens, np.nan)
    if hourly is None:
        for st in np.unique(S.station):
            m = np.where(S.station == st)[0]
            m = m[np.argsort(S.time[m])]
            j = nearest(S.time[m], S.time[m], 12)
            out[m[j >= 0]] = S.dens[m[j[j >= 0]]]
    else:
        o = np.argsort(hourly["time"])
        j = nearest(S.time, hourly["time"][o], 3)
        out[j >= 0] = np.asarray(hourly["dens"], float)[o][j[j >= 0]]
    return out


def mld_skill(pred, S):
    """Mixed-layer depth of the prediction against the observed one, on the samples it covers: mean absolute error,
    bias (predicted minus observed) and correlation (m). A profile that never crosses the threshold gets 195 m."""
    def depth(d):
        m = mld(d)
        return np.where(np.isfinite(np.asarray(d, float)).all(1) & np.isnan(m), DEPTH[-1], m)
    ok = np.isfinite(pred).all(1)
    p, o = depth(pred[ok]), depth(S.dens[ok])
    return dict(mae=float(np.abs(p - o).mean()), bias=float((p - o).mean()), r=float(np.corrcoef(p, o)[0, 1]))


def mode_skill(pred, S, basis):
    """Per-mode R2 of the predicted coefficients on `basis` (a fitted models.Target) and their skill against the
    climatology, 1 - MSE / mean square of the observed coefficient (the climatology predicts 0)."""
    ct, cp = basis.encode(S.anom), basis.encode(pred - S.clim)
    return dict(r2=1 - ((cp - ct) ** 2).sum(0) / ((ct - ct.mean(0)) ** 2).sum(0),
                skill=1 - ((cp - ct) ** 2).mean(0) / (ct ** 2).mean(0))
