"""Horizontal density gradients between pairs of profiles: does the reconstruction put the fronts in the right place?

For two profiles a, b separated by r km the signed gradient at each layer is (rho'_a - rho'_b) / r, with rho' the
density anomaly from the climatology (so the smooth climatological gradient is removed from the observed and the
predicted gradients alike). Scored per layer by
  r      the correlation of the observed and predicted signed gradients (90% interval from a day-block bootstrap)
  sign   the fraction of pairs whose sign agrees
  amp    rms(predicted) / rms(observed), the amplitude
  obs_rms  rms(observed) in g/m3/km
Layers: fixed depths LEVELS, 'ML interior' (mean from the surface to the shallower MLD of the pair) and 'ML base'
(the level of the mean MLD of the pair); the MLDs are the OBSERVED ones.
Pairs: moorings on the same day; same-glider pairs on one continuous transect within a few hours.
"""
import numpy as np
import pandas as pd

from . import data
from .config import DEPTH, N_BOOT
from .evaluate import boot_picks
from .physics import haversine_km, mld

LEVELS = [10, 20, 30, 50, 75, 100, 150]
LAYERS = [f"{z} m" for z in LEVELS] + ["ML interior", "ML base"]
GLIDER_SCALES = [("1-3 km", 1, 3, 6), ("3-7 km", 3, 7, 6), ("7-15 km", 7, 15, 12), ("15-30 km", 15, 30, 24)]
MOORING_SCALES = [("5-15 km", 5, 15), ("15-25 km", 15, 25), ("25-35 km", 25, 35), ("35-60 km", 35, 60)]


# ----------------------------------------------------------------------------- pairs
def segments(glider, lat, lon, dates, min_len=8, dmax_km=7.0, dtmax_h=6.0):
    """Continuous along-track segments per glider (index arrays, time-ordered)."""
    t = pd.DatetimeIndex(dates).values
    out = []
    for g in np.unique(glider):
        m = np.where(glider == g)[0]
        o = m[np.argsort(t[m])]
        dd = haversine_km(lat[o][:-1], lon[o][:-1], lat[o][1:], lon[o][1:])
        dth = np.abs(np.diff(t[o]) / np.timedelta64(1, "h"))
        brk = np.where((dd > dmax_km) | (dth > dtmax_h))[0]
        bnds = np.concatenate(([0], brk + 1, [len(o)]))
        out += [o[a:b] for a, b in zip(bnds[:-1], bnds[1:]) if b - a >= min_len]
    return out


def loiter_mask(glider, lat, lon, dates):
    """True for profiles in segments where the glider loitered (short or isotropic tracks)."""
    loi = np.zeros(len(glider), bool)
    for s in segments(glider, lat, lon, dates):
        x = (lon[s] - lon[s].mean()) * 111.0 * np.cos(np.deg2rad(lat[s].mean()))
        y = (lat[s] - lat[s].mean()) * 111.0
        w = np.linalg.eigvalsh(np.cov(np.column_stack([x, y]).T))
        L1, L2 = np.sqrt(max(w[1], 0)), np.sqrt(max(w[0], 0))
        if L1 * 4 < 5.0 or L1 / max(L2, 1e-9) < 3.0:
            loi[s] = True
    return loi


def glider_pairs(S, lo, hi, dt_h):
    """Same-glider pairs lo <= r < hi km apart within dt_h hours, loiters and off-segment profiles excluded."""
    loi = loiter_mask(S.glider, S.lat, S.lon, S.dates)
    in_seg = np.zeros(S.n, bool)
    for s in segments(S.glider, S.lat, S.lon, S.dates):
        in_seg[s] = True
    t = S.dates.values
    A, B = [], []
    for a in np.where(~loi & in_seg)[0]:
        rr = haversine_km(S.lat[a], S.lon[a], S.lat, S.lon)
        dth = np.abs((t - t[a]) / np.timedelta64(1, "h"))
        sel = np.where((rr >= lo) & (rr < hi) & (dth <= dt_h) & ~loi & in_seg & (np.arange(S.n) > a)
                       & (S.glider == S.glider[a]))[0]
        A += [a] * len(sel)
        B += list(sel)
    A, B = np.array(A, int), np.array(B, int)
    return A, B, haversine_km(S.lat[A], S.lon[A], S.lat[B], S.lon[B])


def mooring_pairs(S, lo, hi):
    """Same-day station pairs lo <= r < hi km apart -> (a, b, separation km)."""
    st = S.station
    pos = {s: data.station_latlon(s) for s in np.unique(st)}
    day = S.dates.normalize()
    A, B, R = [], [], []
    stations = sorted(pos)
    for i, a in enumerate(stations):
        for b in stations[i + 1:]:
            d = float(haversine_km(*pos[a], *pos[b]))
            if not lo <= d < hi:
                continue
            ia = pd.Series(np.where(st == a)[0], index=day[st == a])
            ib = pd.Series(np.where(st == b)[0], index=day[st == b])
            ia, ib = ia[~ia.index.duplicated()], ib[~ib.index.duplicated()]
            common = ia.index.intersection(ib.index)
            A += list(ia[common].values)
            B += list(ib[common].values)
            R += [d] * len(common)
    return np.array(A, int), np.array(B, int), np.array(R, float)


def pairs_by_scale(S):
    if S.name == "gliders":
        return {nm: glider_pairs(S, lo, hi, dt) for nm, lo, hi, dt in GLIDER_SCALES}
    return {nm: mooring_pairs(S, lo, hi) for nm, lo, hi in MOORING_SCALES}


# ----------------------------------------------------------------------------- layers and scores
def layer_values(x, a, b, mld_a, mld_b):
    """(n_pairs, n_layers) of x[a] - x[b] at each layer of LAYERS."""
    diff = x[a] - x[b]
    out = [diff[:, int(np.argmin(np.abs(DEPTH - z)))] for z in LEVELS]
    shallow = np.fmin(mld_a, mld_b)
    interior = np.full(len(a), np.nan)
    for i in np.where(np.isfinite(shallow))[0]:
        sel = DEPTH <= shallow[i]
        if sel.any() and np.isfinite(diff[i, sel]).any():
            interior[i] = np.nanmean(diff[i, sel])
    out.append(interior)
    base = (mld_a + mld_b) / 2
    ib = np.where(np.isfinite(base), np.abs(DEPTH[None, :] - np.nan_to_num(base)[:, None]).argmin(1), 0)
    out.append(np.where(np.isfinite(base), diff[np.arange(len(a)), ib], np.nan))
    return np.column_stack(out)


def _corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan


def score(gt, gp, picks):
    ok = np.isfinite(gt) & np.isfinite(gp)
    if ok.sum() < 20:
        return None
    bs = [_corr(gt[p][ok[p]], gp[p][ok[p]]) for p in picks]
    lo, hi = np.nanpercentile(bs, [5, 95])
    return dict(n=int(ok.sum()), r=_corr(gt[ok], gp[ok]), r_lo=float(lo), r_hi=float(hi),
                sign=float(np.mean(np.sign(gt[ok]) == np.sign(gp[ok]))),
                amp=float(np.sqrt(np.mean(gp[ok] ** 2) / np.mean(gt[ok] ** 2))),
                obs_rms=float(1000 * np.sqrt(np.mean(gt[ok] ** 2))))


def paired_diff(gt, ga, gb, picks):
    """r(a) - r(b) on the pairs both fields cover, with its bootstrap 90% interval."""
    ok = np.isfinite(gt) & np.isfinite(ga) & np.isfinite(gb)

    def stat(p):
        f = p[ok[p]]
        return _corr(gt[f], ga[f]) - _corr(gt[f], gb[f])
    bs = np.array([stat(p) for p in picks])
    return stat(np.arange(len(gt))), float(np.nanpercentile(bs, 5)), float(np.nanpercentile(bs, 95))


def gradient_scores(S, fields, compare=("SWOT", "DUACS"), n_boot=N_BOOT):
    """fields: {name: density (n, 40) at test set S (moorings or gliders)}. One row per (scale, layer, field); when
    both `compare` fields are given, dr = r(compare[0]) - r(compare[1]) with its interval."""
    obs = S.anom
    fields = {k: v - S.clim for k, v in fields.items()}
    mlds = mld(S.dens)
    rows = []
    for scale, (a, b, rk) in pairs_by_scale(S).items():
        if len(a) < 20:
            continue
        blocks = S.dates[a].normalize().values.astype("datetime64[D]").astype(int)
        picks = boot_picks(blocks, n_boot)
        gt = layer_values(obs, a, b, mlds[a], mlds[b]) / rk[:, None]
        gp = {k: layer_values(v, a, b, mlds[a], mlds[b]) / rk[:, None] for k, v in fields.items()}
        for j, L in enumerate(LAYERS):
            dr = (paired_diff(gt[:, j], gp[compare[0]][:, j], gp[compare[1]][:, j], picks)
                  if len(compare) == 2 and all(c in gp for c in compare) else (np.nan, np.nan, np.nan))
            for k in fields:
                sc = score(gt[:, j], gp[k][:, j], picks)
                if sc is None:
                    continue
                rows.append(dict(platform=S.name, scale=scale, layer=L, field=k, n_pairs=len(a),
                                 n_days=len(np.unique(blocks)), **sc, dr=dr[0], dr_lo=dr[1], dr_hi=dr[2]))
    return pd.DataFrame(rows)
