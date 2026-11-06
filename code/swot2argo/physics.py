"""Static stability, mixed-layer depth, distances."""
import numpy as np

from .config import DEPTH


def _pava(y):
    """Nearest non-decreasing sequence to y in the least-squares sense (pool-adjacent-violators)."""
    sums, cnts = [], []
    for v in y:
        s, c = float(v), 1
        while sums and sums[-1] / cnts[-1] > s / c:
            s += sums.pop()
            c += cnts.pop()
        sums.append(s)
        cnts.append(c)
    return np.repeat(np.array(sums) / np.array(cnts), cnts)


def stabilize(dens):
    """Project total density profiles (n, 40) onto the statically stable ones (non-decreasing with depth).
    A projection: stable profiles are unchanged, and the RMSE to any stable profile can only decrease."""
    dens = np.asarray(dens, float)
    out = dens.reshape(-1, dens.shape[-1]).copy()
    unstable = np.any(np.diff(out, axis=1) < 0, axis=1) & np.isfinite(out).all(1)
    for i in np.where(unstable)[0]:
        out[i] = _pava(out[i])
    return out.reshape(dens.shape)


def mld(dens, dsig=0.125, ref=10.0):
    """Mixed-layer depth: where density first exceeds its value at `ref` m by `dsig` kg/m3 (NaN if never)."""
    dens = np.atleast_2d(dens)
    i10 = int(np.argmin(np.abs(DEPTH - ref)))
    over = dens > dens[:, [i10]] + dsig
    over[:, :i10 + 1] = False
    first = np.where(over.any(1), over.argmax(1), -1)
    return np.where(first >= 0, DEPTH[np.clip(first, 0, None)], np.nan)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.deg2rad(np.asarray(x, float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
