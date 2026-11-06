"""The figures of the paper and of its supporting information. Each function draws one figure and returns it;
`save(fig, name)` writes it to figures/ at 300 dpi.

    overview(tests, basis_anom, swath)            Figure 1: maps, the hourly S2 record, the PCA basis
    disagreement_examples(preds, tests, cases)    Figure 2: overpasses on which SWOT and DUACS differ at the site
    season_gain(preds, tests, D, crossovers)      Figure 3: when SWOT input is ahead of DUACS input
    swath_compare(S, G, sources, tests)           Figure 4: one overpass reconstructed in 3-D, section vs moorings
    pca_gradient_modes(tests, basis_anom)         Figure S1: the PCA reconstruction in anomaly space, and the modes
                                                  that carry the horizontal gradients
    s2_daily(preds, tests, bench)                 Figure S2: every S2 profile of the science-orbit year
    gradients_by_scale(preds, tests, bench)       Figure S3: horizontal gradient magnitude and its distribution
    gliders(preds, tests, D, bench)               Figure S4: glider transects and the gradient skill along them
    tide_lag(preds, tests, pairs, oof)            Figure S5: SWOT's gain against the time between profile and pass

Style: matplotlib defaults at font size 11; observed in black, the network with SWOT input in red, with DUACS input
in blue dash-dot, the climatology in green dotted, GLORYS12 in purple dashed; cmocean colour maps.
"""
import os

import cmocean
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator

from . import data
from .config import DEPTH, FIGURES_DIR
from .evaluate import boot_picks
from .gradients import GLIDER_SCALES, LAYERS, MOORING_SCALES, layer_values, loiter_mask, mooring_pairs, segments
from .physics import haversine_km, mld

plt.rcParams.update({"font.size": 11})

TRUE_C, PRED_C, DUACS_C, CLIM_C = "#000000", "#D7191C", "#0072B2", "#009E73"
BENCH_C = {"glorys12": "#7B3294", "hycom": "#56B4E9", "armor3d": "#E69F00", "glo12": "#8C510A"}
BENCH_LS = (0, (4, 2))
BENCH_SHORT = {"glorys12": "GLORYS12", "hycom": "HYCOM", "armor3d": "ARMOR3D", "glo12": "GLO12"}
DPI = 300
Z = -DEPTH                                           # negative depth: 0 at the top, -195 at the bottom
YLIM = (-200, 0)
MONTHS = [4, 5, 6, 7]
LEGEND = [Line2D([0], [0], color=TRUE_C, lw=2.2, label="True"),
          Line2D([0], [0], color=PRED_C, lw=2.2, label="Prediction (SWOT)"),
          Line2D([0], [0], color=DUACS_C, lw=1.8, ls="-.", label="Prediction (DUACS)"),
          Line2D([0], [0], color=CLIM_C, lw=2.2, ls=":", label="Climatology")]


def save(fig, name, dpi=DPI):
    path = os.path.join(FIGURES_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=getattr(fig, "save_bbox", "tight"))
    plt.close(fig)
    print("wrote", path)
    return path


def _panel_letter(ax, letter, x=-0.16, y=1.06, ha="left"):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha=ha)


# ============================================================================= Figure 1
def _coast_and_swaths(ax, G):
    """The coastline and the SWOT swath edges, the background of every map."""
    ax.fill(G["coast"][:, 0], G["coast"][:, 1], color="0.85", ec="0.5", lw=0.5, zorder=1)
    for key in [key for key in G if key.startswith("swath_") and key.endswith("_lat")]:
        ax.plot(G[key.replace("_lat", "_lon")], G[key], color="0.6", lw=0.4, alpha=0.7, zorder=2)


def _moorings(ax, G, labels=True, big=True, dx=(0.05, -0.2)):
    """The 11 moorings: S in red, P in orange, S2 a star."""
    for st, la, lo in zip(G["stations"], G["station_lat"], G["station_lon"]):
        s2 = st == "S2"
        ax.plot(lo, la, "*" if s2 else "o", color="r" if st.startswith("S") else "orange",
                ms=(12 if big else 8) if s2 else (6 if big else 4), mec="k", mew=0.4, zorder=6)
        if labels:
            right = st in ("P1", "P2", "P3", "P4", "S1", "S2")
            ax.text(lo + (dx[0] if right else dx[1]), la + (0.01 if right else -0.05), st, fontsize=9, zorder=7)


def _map_frame(ax, box):
    la0, la1, lo0, lo1 = box
    ax.set_xlim(lo0, lo1)
    ax.set_ylim(la0, la1)
    ax.set_aspect(1 / np.cos(np.deg2rad(36)))
    ax.grid(alpha=0.3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


MOORING_LEGEND = [Line2D([0], [0], marker="o", ls="", color="r", mec="k", mew=0.4, label="S-moorings"),
                  Line2D([0], [0], marker="o", ls="", color="orange", mec="k", mew=0.4, label="P-moorings"),
                  Line2D([0], [0], marker="*", ls="", color="r", mec="k", mew=0.4, ms=10,
                         label="S2 (Mar 2023 - Oct 2024)")]


def _map_panels(axes):
    """The region (axes[0]) with the training pairs, and the array close-up (axes[1]). Returns the legend handles."""
    G, P = data.geo(), data.pairs()
    boxes = [(G["box_lat"][0], G["box_lat"][1], G["box_lon"][0], G["box_lon"][1]), (34.6, 37.2, -126.6, -123.4)]
    for k, (ax, box) in enumerate(zip(axes, boxes)):
        _coast_and_swaths(ax, G)
        ax.scatter(P.lon, P.lat, s=14, c="0.45", zorder=4, label=f"Argo training pairs (n={P.n})")
        _moorings(ax, G, labels=bool(k), big=bool(k))
        _map_frame(ax, box)
    axes[0].add_patch(Rectangle((-126.6, 34.6), 3.2, 2.6, fill=False, edgecolor="r", lw=1.2, zorder=8))
    h, _ = axes[0].get_legend_handles_labels()
    return h + MOORING_LEGEND


def swath_background(ax, fname, tests, alpha=0.5):
    """One overpass's calibrated SWOT SSHA under a map (cmocean delta, symmetric about zero at the 99.5th percentile
    of |SSHA| over the half-swath holding the array). Returns the mesh, for a colour bar."""
    from matplotlib.colors import ListedColormap, Normalize
    G = data.swath(fname)
    S = data.swath_pixels(G)
    valid = np.zeros(G["lat"].shape, bool)
    valid[S.line, S.pixel] = True
    g = _swath_geometry(valid, G, tests["moorings"])
    smax = np.nanpercentile(np.abs(np.concatenate([t[:, g.cols].ravel() for t in (G["ssh_swot"], G["ssh_duacs"])])), 99.5)
    rgb = cmocean.cm.delta(np.linspace(0, 1, 256))[:, :3]          # the opacity is baked into the colour map
    light = ListedColormap(1 - alpha * (1 - rgb), name="delta_light")
    return ax.pcolormesh(G["lon"], G["lat"], np.ma.masked_invalid(G["ssh_swot"]), cmap=light,
                         norm=Normalize(-smax, smax), shading="nearest", rasterized=True, zorder=2.5)


def _hovmoller(ax, cax, max_gap_h=3):
    """Hourly S2 density against time and depth with the mixed-layer depth; gaps in the record are left blank."""
    H = data.s2_hourly()
    t, rho = pd.DatetimeIndex(H["time"]), H["dens"].astype(float)
    gap = np.where(np.diff(t.values) > np.timedelta64(max_gap_h, "h"))[0]
    hour = np.timedelta64(1, "h")
    t_pad = np.insert(t.values, np.repeat(gap + 1, 2), np.ravel([[t.values[i] + hour, t.values[i + 1] - hour] for i in gap]))
    rho_pad = np.insert(rho, np.repeat(gap + 1, 2), np.nan, axis=0)
    pc = ax.pcolormesh(t_pad, Z, rho_pad.T, cmap=cmocean.cm.dense, shading="nearest", rasterized=True)
    ax.plot(t_pad, -mld(rho_pad), color="k", lw=0.6, label="S2 MLD (hourly)")
    ax.set_ylim(*YLIM)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Depth (m)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    ax.figure.colorbar(pc, cax=cax).set_label("S2 density (kg/m³)")


def _recon_panels(top, tgt, tests, station="S1", anomaly=False):
    """One profile of `station` near the 15th of each campaign month (observed, black) and its projection on the
    first 2 (light grey dotted) and 4 (dark grey dashed) modes of the basis `tgt`; as density (Figure 1d) or, with
    anomaly=True, as the anomaly from the climatology, the space of the PCA (Figure S1)."""
    S = tests["moorings"]
    rows = np.where(S.station == station)[0]

    def recon(a, k):
        M = tgt.all_modes[:k]
        return (a - tgt.mean) @ M.T @ M + tgt.mean

    for ax, m in zip(top, MONTHS):
        target = pd.Timestamp(2023, m, 15)
        i = rows[np.argmin(np.abs((S.dates[rows] - target).values))]
        base = 0.0 if anomaly else S.clim[i]
        ax.plot(S.anom[i] if anomaly else S.dens[i], Z, color="k", lw=2, label=f"True {station}")
        ax.plot(base + recon(S.anom[i], 4), Z, color="0.35", ls="--", lw=2.2, label="PCA recon. (4 modes)")
        ax.plot(base + recon(S.anom[i], 2), Z, color="0.65", ls=":", lw=2.4, label="PCA recon. (2 modes)")
        if anomaly:
            ax.axvline(0, color="0.6", lw=0.8)
        else:
            ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_title(S.dates[i].strftime("%-d %B %Y"))
        ax.set_xlabel("Density anomaly (kg/m³)" if anomaly else "Density (kg/m³)")
        ax.set_ylim(*YLIM)
        ax.grid(True, alpha=0.3)
        if ax is top[0]:
            ax.set_ylabel("Depth (m)")
            ax.legend(loc="lower left", fontsize=8)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    return top


def _basis_panels(fig, gs, basis_anom, tests, station="S1", n_show=4):
    """Fill a 2 x 13 gridspec: reconstructions of one mooring with 2 and 4 modes, then the basis itself (the mean
    anomaly, the modes, the cumulative variance). The basis is fitted on `basis_anom` (the pretraining anomalies)."""
    from .models import Target
    tgt = Target(None).fit(basis_anom)
    gt = gs[0, :].subgridspec(1, 4, wspace=0.12)
    top = [fig.add_subplot(gt[0, i]) for i in range(4)]
    _recon_panels(top, tgt, tests, station)
    ax0 = fig.add_subplot(gs[1, 0:4])
    ax0.plot(tgt.mean, Z, "k-", lw=2)
    ax0.set_xlabel("Density Anomaly (kg/m³)")
    ax0.set_ylabel("Depth (m)")
    ax0.set_title("Mean Anomaly Profile")
    ax0.set_ylim(*YLIM)
    ax0.grid(True, alpha=0.3)
    ax1 = fig.add_subplot(gs[1, 4:8], sharey=ax0)
    for i, ls in enumerate(["-", "--", "-.", ":"][:n_show]):
        ax1.plot(tgt.all_modes[i], Z, lw=2, ls=ls, label=f"Mode {i + 1} ({tgt.explained[i]:.0%})")
    ax1.set_xlabel("Amplitude")
    ax1.set_title("PCA Modes")
    ax1.legend(loc="lower right", ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax2 = fig.add_subplot(gs[1, 9:13])
    cum = np.cumsum(tgt.explained)
    ax2.step(range(1, len(cum) + 1), cum, lw=2, where="mid", color="k")
    ax2.axvline(4, color="k", ls="--", lw=1)
    ax2.set_xticks([1, 4, 10, 20, 30, len(cum)])
    ax2.set_xlim(0, len(cum))
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Variance Explained")
    ax2.set_title("Cumulative PCA Variance")
    ax2.grid(True, alpha=0.3)
    return top + [ax0, ax1, ax2]


def overview(tests, basis_anom, swath, station="S1", hov_up_cm=0.6):
    """Figure 1: the maps (a, b; `swath` drawn as the SSHA background of b), the hourly S2 record (c), one mooring's
    profiles on the PCA basis (d) and the basis itself (e-g), fitted on `basis_anom`."""
    fig = plt.figure(figsize=(10.5, 11.5))
    fig.subplots_adjust(top=0.97, bottom=0.04)
    outer = fig.add_gridspec(3, 1, height_ratios=[4.4, 1.3, 4.0], hspace=0.17)
    gm = outer[0].subgridspec(2, 3, width_ratios=[1, 1.075, 0.035], height_ratios=[3.55, 0.3], wspace=0.28, hspace=0.26)
    maps = [fig.add_subplot(gm[0, 0]), fig.add_subplot(gm[0, 1])]
    h = _map_panels(maps)
    maps[0].set_title("Training Data: Argo under SWOT (2023-2026)", fontsize=11)
    maps[1].set_title("Test Data: the Cal/Val Array (Apr-Jul 2023)", fontsize=11)
    maps[0].xaxis.set_major_locator(MultipleLocator(4))
    maps[1].xaxis.set_major_locator(MultipleLocator(1))
    maps[1].set_ylabel("")
    m_ssh = swath_background(maps[1], swath, tests)
    cb = fig.colorbar(m_ssh, cax=maps[1].inset_axes([1.04, 0, 0.04, 1]))
    cb.set_label("SSHA (m)")
    cb.ax.yaxis.set_major_locator(MultipleLocator(0.05))
    lax = fig.add_subplot(gm[1, :])
    lax.set_axis_off()
    lax.legend(handles=h, loc="center", ncol=4, fontsize=11, frameon=True, framealpha=0.9, columnspacing=1.2,
               handletextpad=0.4)
    gh = outer[1].subgridspec(1, 2, width_ratios=[1, 0.018], wspace=0.03)
    hov, hcax = fig.add_subplot(gh[0, 0]), fig.add_subplot(gh[0, 1])
    _hovmoller(hov, hcax)
    for ax in (hov, hcax):                                # the S2 record raised towards the maps
        bb = ax.get_position()
        ax.set_position([bb.x0, bb.y0 + hov_up_cm / 2.54 / fig.get_figheight(), bb.width, bb.height])
    gb = outer[2].subgridspec(2, 13, height_ratios=[1.1, 1], hspace=0.65, wspace=1.6)
    basis = _basis_panels(fig, gb, basis_anom, tests, station)
    for ax, letter in zip(maps + [hov, basis[0]] + basis[4:], "abcdefg"):
        ax.annotate(letter, (0, 1), xytext=(-30, 7), xycoords="axes fraction", textcoords="offset points",
                    fontweight="bold", fontsize=13, ha="left", va="bottom")
    return fig


# ============================================================================= Figure 2
CASES = [("s2", "swath_s2_20240318_11.npz", "S2"),               # the two inputs agree (spring)
         ("moorings", "swath_20230530_22.npz", "P4"),             # they differ: spring
         ("moorings", "swath_20230705_04.npz", "P3"),             # summer
         ("s2", "swath_s2_20241002_17.npz", "S2")]                # autumn
SSHA_VMAX_CM = 12                                                 # the colour scale of the maps
SSHA_LEVELS_CM = np.arange(-40, 41, 2)                            # the contours, the same on both maps
SWOT_SMOOTH_PX = 2                                                # 4 km: the Gaussian under SWOT's contours only
SSHA_CMAP = cmocean.tools.crop_by_percent(cmocean.cm.delta, 12, which="both")   # dark ends off: contours stay visible


def _nan_smooth(F, sigma):
    """Gaussian smoothing (sigma in pixels, kernel to 4 sigma, ends mirrored) that leaves the holes of the swath empty
    and does not spread them: the smoothed field divided by the smoothed mask of valid pixels."""
    r = int(4 * sigma + 0.5)
    x = np.arange(-r, r + 1)
    w = np.exp(-0.5 / sigma ** 2 * x ** 2)
    w /= w.sum()

    def blur(A):
        for axis in (0, 1):
            A = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="symmetric"), w, "valid"), axis, A)
        return A
    ok = np.isfinite(F)
    return np.where(ok, blur(np.where(ok, F, 0.0)) / blur(ok.astype(float)), np.nan)


def disagreement_examples(preds, tests, cases=CASES, half=0.5):
    """Figure 2: one column per overpass (test set, swath file, station). Rows: the SWOT SSHA around the site, the
    DUACS SSHA at the same pixels (each minus its mean over the window, black contours every 2 cm, SWOT's on a copy
    smoothed over 4 km), and the density profile at the site, observed and reconstructed with each input, with their
    MLDs."""
    fig = plt.figure(figsize=(2.45 * len(cases), 9.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[2, 1.35], hspace=0.3)
    gmap, gprof = outer[0].subgridspec(2, len(cases), hspace=0.15), outer[1].subgridspec(1, len(cases))
    axes = np.array([[fig.add_subplot(gmap[r, c]) for c in range(len(cases))] for r in range(2)]
                    + [[fig.add_subplot(gprof[0, c]) for c in range(len(cases))]])
    for c, (nm, f, station) in enumerate(cases):
        S = tests[nm]
        G = data.swath(f)
        t = pd.Timestamp(G["time"])
        m = np.abs((S.dates - t).total_seconds()) < 3 * 3600
        if "station" in S:
            m &= S.station == station
        k = int(np.where(m)[0][0])
        la, lo = float(S.lat[k]), float(S.lon[k])
        for r, key in enumerate(("ssh_swot", "ssh_duacs")):
            ax = axes[r, c]
            win = (np.abs(G["lon"] - lo) < half) & (np.abs(G["lat"] - la) < half * 1.15)
            if key == "ssh_swot":          # calibrated but not outlier-screened: no holes in the strong highs
                i0 = np.unravel_index(np.nanargmin((G["lat"] - la) ** 2 + (G["lon"] - lo) ** 2), G["lat"].shape)
                jj, ii = np.meshgrid(np.arange(G["lat"].shape[1]) - i0[1], np.arange(G["lat"].shape[0]) - i0[0])
                field = G["ssh_raw"] - (G["tilt"][0] + G["tilt"][1] * jj + G["tilt"][2] * ii)
            else:
                field = G[key]
            F = (field - np.nanmean(field[win])) * 100
            pc = ax.pcolormesh(G["lon"], G["lat"], F, cmap=SSHA_CMAP, vmin=-SSHA_VMAX_CM, vmax=SSHA_VMAX_CM,
                               shading="nearest")
            ax.contour(G["lon"], G["lat"], _nan_smooth(F, SWOT_SMOOTH_PX) if key == "ssh_swot" else F,
                       levels=SSHA_LEVELS_CM, colors="k", linewidths=0.55, linestyles="solid")
            ax.plot(lo, la, "^", ms=8, c="k", mec="w", mew=0.8, zorder=5)
            ax.set_xlim(lo - half, lo + half)
            ax.set_ylim(la - half * 1.15, la + half * 1.15)
            ax.set_aspect(1 / np.cos(np.deg2rad(la)))
            ax.set_xticks(np.round([lo - half / 2, lo + half / 2], 1))
            ax.set_yticks(np.round([la - half / 2, la + half / 2], 1))
            ax.tick_params(labelsize=8)
            if c > 0:
                ax.set_yticklabels([])
            if r == 0:
                d = (S.ssh_swot[k] - S.ssh_duacs[k]) * 100
                ax.set_title(f"{t:%-d %B %Y}, {station}\n$\\Delta$SSHA = {d:+.1f} cm", fontsize=9)
                ax.set_xticklabels([])
            ax.text(0.06, 0.95, ("SWOT", "DUACS")[r], transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(fc="w", ec="0.6", lw=0.5, alpha=0.9, pad=2.5))
        ax = axes[2, c]
        ps, pd_ = preds["swot"][nm][k], preds["duacs"][nm][k]
        ax.plot(S.dens[k], Z, color=TRUE_C, lw=2.2)
        ax.plot(ps, Z, color=PRED_C, lw=2.2)
        ax.plot(pd_, Z, color=DUACS_C, lw=1.8, ls="-.")
        for prof, col, ls in ((S.dens[k], TRUE_C, "-"), (ps, PRED_C, "-"), (pd_, DUACS_C, "-.")):
            z = mld(prof)[0]
            if np.isfinite(z):
                ax.axhline(-z, color=col, lw=1.1, ls=ls, alpha=0.9)
        ax.set_ylim(*YLIM)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Density (kg/m³)")
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        if c == 0:
            ax.set_ylabel("Depth (m)")
        else:
            ax.set_yticklabels([])
    for r in range(2):
        axes[r, 0].set_ylabel("Latitude")
    for ax in axes[1]:
        ax.set_xlabel("Longitude", fontsize=9)
    fig.canvas.draw()                                     # the colour bar spans the two map rows as drawn
    boxes = [ax.get_position() for ax in axes[:2].ravel()]
    y0, y1, x1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes), max(b.x1 for b in boxes)
    cb = fig.colorbar(pc, cax=fig.add_axes([x1 + 0.012, y0, 0.012, y1 - y0]), extend="both",
                      ticks=np.arange(-SSHA_VMAX_CM, SSHA_VMAX_CM + 1, 4))
    cb.set_label("SSHA minus its mean over the window (cm)")
    handles = [Line2D([], [], color=TRUE_C, lw=2.2, label="Observed"),
               Line2D([], [], color=PRED_C, lw=2.2, label="Prediction (SWOT)"),
               Line2D([], [], color=DUACS_C, lw=1.8, ls="-.", label="Prediction (DUACS)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01), frameon=True)
    return fig


# ============================================================================= Figure 3
NOISE_C, FEATURE_C = "0.45", "#E69F00"
S2_YEAR = (pd.Timestamp("2023-09-25"), pd.Timestamp("2024-11-05"))      # the time axis of a and b: the S2 record
MOOR_X = (pd.Timestamp("2023-03-31"), pd.Timestamp("2023-07-30"))       # of c and d, drawn under April-July of b


def _gain(es, ed):
    """RMSE with DUACS input minus RMSE with SWOT input, from the mean squares of each profile: above 0, SWOT is
    better."""
    return np.sqrt(ed.mean()) - np.sqrt(es.mean())


def _wcorr(x, y, w):
    """Correlation of x and y with weights w."""
    cx, cy = x - (w * x).sum() / w.sum(), y - (w * y).sum() / w.sum()
    return (w * cx * cy).sum() / np.sqrt((w * cx * cx).sum() * (w * cy * cy).sum())


def _crossover_season(C, n=1000, seed=0):
    """SWOT - DUACS on the swath where two passes less than 2 days apart cross (C = data.swath_crossovers()), split
    into the covariance of the two passes (correlated: features DUACS misses that outlast the gap) and the rest of
    the variance (uncorrelated: mostly SWOT noise). Each crossing's mean is removed and the grand mean added back, so
    the changing mix of crossings cannot make a season. Returns the mean, lo, hi (12 calendar months x [uncorrelated,
    correlated], cm2) in centred 3-month pools over all years, with 90 % intervals from n resamples of SWOT cycles."""
    X = pd.DataFrame(dict(noise=C.var_cm2 - C.cov_cm2, feature=C.cov_cm2))
    place = C.pass_a.astype(str) + "_" + C.pass_b.astype(str) + "_" + C.lat.round(2).astype(str)
    V = (X - X.groupby(place).transform("mean") + X.mean()).to_numpy()
    month = C.time.dt.month.to_numpy()

    def stat(i):
        return np.array([V[i][np.isin(month[i], ((m - 2) % 12 + 1, m, m % 12 + 1))].mean(0) for m in range(1, 13)])
    cycles = np.unique(C.cycle_a)
    rows = {c: np.where(C.cycle_a == c)[0] for c in cycles}
    rng = np.random.default_rng(seed)
    bs = [stat(np.concatenate([rows[c] for c in rng.choice(cycles, len(cycles))])) for _ in range(n)]
    return (stat(np.arange(len(V))), *np.nanpercentile(bs, [5, 95], axis=0))


def _window3_gain(es, ed, t, n=1000, seed=0):
    """The RMSE gain in a centred 3-month window at every month of the record (neighbouring windows share passes),
    with 90 % intervals from n resamples of the passes. Returns (mid-month times, value, lo, hi)."""
    per = pd.DatetimeIndex(t).to_period("M")
    rng = np.random.default_rng(seed)
    out = []
    for m in pd.period_range(per.min(), per.max(), freq="M"):
        sel = np.where((per >= m - 1) & (per <= m + 1))[0]
        bs = [_gain(es[i], ed[i]) for i in (rng.choice(sel, len(sel)) for _ in range(n))]
        out.append((pd.Timestamp(m.year, m.month, 15), _gain(es[sel], ed[sel]), *np.percentile(bs, [5, 95])))
    x, v, lo, hi = zip(*out)
    return pd.DatetimeIndex(x), np.array(v), np.array(lo), np.array(hi)


def _monthly_gain(es, ed, S, n=1000):
    """The RMSE gain at the moorings by month, with 90 % intervals from n block-bootstrap resamples (station x week).
    Returns (mid-month times, value, lo, hi)."""
    picks = boot_picks(data.bootstrap_blocks(S), n)
    out = []
    for m in MONTHS:
        sel = np.asarray(S.dates.month == m)
        bs = [_gain(es[q], ed[q]) for q in (p[sel[p]] for p in picks) if len(q) > 2]
        out.append((pd.Timestamp(2023, m, 15), _gain(es[sel], ed[sel]), *np.nanpercentile(bs, [5, 95])))
    x, v, lo, hi = zip(*out)
    return pd.DatetimeIndex(x), np.array(v), np.array(lo), np.array(hi)


def _gradient_gain(S, pred_swot, pred_duacs, n=1000, seed=0):
    """Correlation of the observed signed mixed-layer gradient between mooring pairs with the one predicted with SWOT
    input minus with DUACS input, by month, one entry per separation class, with 90 % intervals from n resamples of
    whole days. As in the paper's figure, the observed gradient is that of the anomaly from the climatology and the
    predicted ones are those of the density. Returns [(mid-month times, value, lo, hi)]."""
    ml = mld(S.dens)
    j = LAYERS.index("ML interior")
    out = []
    for _, lo, hi in MOORING_SCALES:
        a, b, rk = mooring_pairs(S, lo, hi)
        keep = np.isfinite(S.dens[a]).all(1) & np.isfinite(S.dens[b]).all(1)
        a, b, rk = a[keep], b[keep], rk[keep]
        g = np.array([layer_values(x, a, b, ml[a], ml[b])[:, j] / rk for x in (S.anom, pred_swot, pred_duacs)])
        ok = np.isfinite(g).all(0)
        gt, gs, gd = g[:, ok]
        t = S.dates[a[ok]]
        mo, months = t.month.values, np.unique(t.month.values)
        day = t.normalize().values.astype("datetime64[D]").astype(int)

        def stat(w):
            return np.array([_wcorr(gt[k], gs[k], w[k]) - _wcorr(gt[k], gd[k], w[k])
                             for k in (mo == m for m in months)])
        u = np.unique(day)
        pos = np.searchsorted(u, day)
        rng = np.random.default_rng(seed)
        bs = [stat(np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[pos].astype(float)) for _ in range(n)]
        out.append((pd.DatetimeIndex([pd.Timestamp(2023, m, 15) for m in months]), stat(np.ones(len(gt))),
                    *np.nanpercentile(bs, [5, 95], axis=0)))
    return out


def _draw_ci(ax, x, v, lo, hi, star_pad):
    """Black points with 90 % intervals joined by a line, an asterisk where the interval is above zero."""
    ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="o-", color="k", ms=6, lw=1.6, capsize=3, elinewidth=1.2)
    for xi, h, l in zip(x, hi, lo):
        if l > 0:
            ax.text(xi, h + star_pad, "*", ha="center", va="bottom", fontsize=14)
    ax.axhline(0, color="0.5", lw=0.8)


def _year_axis(ax):
    """The months of the S2 record, October 2023 to October 2024, one tick at the middle of each."""
    def lab(x, pos=None):
        d = mdates.num2date(x)
        return f"{d:%b}" + (f"\n{d:%Y}" if d.month == 1 or (d.month == 10 and d.year == 2023) else "")
    ax.set_xlim(*S2_YEAR)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15))
    ax.xaxis.set_major_formatter(lab)
    ax.grid(alpha=0.3)


def _moorings_axis(ax, labels):
    """April to July 2023, one tick at the middle of each month."""
    ax.set_xlim(*MOOR_X)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15))
    ax.xaxis.set_major_formatter(lambda x, pos=None: f"{mdates.num2date(x):%b}"
                                 + ("\n2023" if mdates.num2date(x).month == 4 else ""))
    ax.tick_params(labelbottom=labels)
    ax.grid(alpha=0.3)


def _gradient_bars(axes, D, labels):
    """e: the correlation of the signed mixed-layer gradient over April-July (D = data.gradients()), one axis per
    separation: SWOT input, DUACS input, GLORYS12 from the top, 90 % intervals, * where SWOT minus DUACS has its
    interval above zero."""
    sub = D[(D.platform == "moorings") & (D.layer == "ML interior")]
    fields = ((("network", "SWOT"), PRED_C), (("network", "DUACS"), DUACS_C),
              (("glorys12", "analysis"), BENCH_C["glorys12"]))
    for k, (ax, (sc, _, _)) in enumerate(zip(axes, MOORING_SCALES)):
        for j, ((model, field), col) in enumerate(fields):
            q = sub[(sub.model == model) & (sub.field == field) & (sub.scale == sc)].iloc[0]
            ax.barh(2 - j, q.r, 0.8, color=col)
            ax.errorbar(q.r, 2 - j, xerr=[[q.r - q.r_lo], [q.r_hi - q.r]], fmt="none", ecolor="k", lw=0.9, capsize=2)
            if j == 0 and q.dr_lo > 0:
                ax.text(q.r_hi + 0.02, 2, "*", ha="left", va="center", fontsize=14)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([1])
        ax.set_yticklabels([labels[k]])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 0.8)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.tick_params(labelbottom=k == len(axes) - 1)
        ax.grid(axis="x", alpha=0.3)
    axes[-1].set_xlabel("Correlation")


def season_gain(preds, tests, D, crossovers):
    """Figure 3: (a) SWOT - DUACS on the swath through the mean year, split at crossing passes into an uncorrelated
    and a correlated part (crossovers = data.swath_crossovers()); the profile gain, RMSE with DUACS input minus with
    SWOT input over 0-200 m, (b) at S2 in centred 3-month windows and (c) at the moorings by month, under April-July
    of b; (d) the correlation of the signed mixed-layer gradient between mooring pairs with SWOT input minus with
    DUACS input, by month, one row per separation, and (e) each input and GLORYS12 over April-July (D =
    data.gradients()). 90 % intervals; * where the interval is above zero."""
    S, S2 = tests["moorings"], tests["s2"]
    sq = lambda name, T: ((preds[name][T.name] - T.dens) ** 2).mean(1)                     # noqa: E731
    W, L = 9.0, 1.25
    AW = W - L - 0.3
    rows = [("gap", 0.4), ("a", 1.8), ("gap", 0.4), ("b", 1.8), ("gap", 1.0), ("c", 1.35)]
    rows += [x for k in range(4) for x in (("gap", 0.95 if k == 0 else 0.22), (f"d{k}", 0.95))]
    rows += [("gap", 0.75)]
    H = sum(h for _, h in rows)
    top, y = {}, 0.0                                     # each row's distance from the top and its height (inches)
    for k, h in rows:
        top[k] = (y, h)
        y += h
    fig = plt.figure(figsize=(W, H))

    def axes_at(key, x0=L, w=AW, **kw):
        y0, h = top[key]
        return fig.add_axes([x0 / W, (H - y0 - h) / H, w / W, h / H], **kw)

    def xpos(t):
        return L + AW * (t - S2_YEAR[0]) / (S2_YEAR[1] - S2_YEAR[0])

    def text_at(x_in, y_in, txt, **kw):
        fig.text(x_in / W, 1 - y_in / H, txt, **kw)

    # (a) the swath through the mean year, drawn on the months of the S2 record
    ax_a = axes_at("a")
    v, lo, hi = _crossover_season(crossovers)
    months = [10, 11, 12] + list(range(1, 11))
    t = [pd.Timestamp(2023 if i < 3 else 2024, m, 15) for i, m in enumerate(months)]
    k = np.array(months) - 1
    for j, (col, ls, lab) in enumerate(((NOISE_C, "--", "Uncorrelated: SWOT noise"),
                                        (FEATURE_C, "-", "Correlated: features missed by DUACS"))):
        ax_a.fill_between(t, lo[k, j], hi[k, j], color=col, alpha=0.2, lw=0)
        ax_a.plot(t, v[k, j], color=col, ls=ls, lw=2, label=lab)
    _year_axis(ax_a)
    ax_a.set_ylim(np.floor(lo.min() * 10) / 10, np.ceil(hi.max() * 10) / 10 + 0.2)     # room for the legend
    ax_a.set_ylabel("Variance (cm$^2$)")
    ax_a.set_title("SWOT $-$ DUACS on the swath, mean year of 2023-2026", fontsize=11)
    ax_a.tick_params(labelbottom=False)
    ax_a.legend(loc="upper right", title="Between crossing passes", title_fontsize=9, alignment="left", fontsize=9,
                frameon=False, handlelength=2.2)
    # (b) S2
    ax_b = axes_at("b", sharex=ax_a)
    x, v, lo, hi = _window3_gain(sq("swot", S2), sq("duacs", S2), S2.dates)
    _draw_ci(ax_b, x, v, lo, hi, 0.002)
    ax_b.set_ylim(min(lo.min(), 0) - 0.006, hi.max() + 0.015)
    ax_b.set_ylabel("RMSE gain\n(kg m$^{-3}$)")
    ax_b.legend(handles=[Line2D([0], [0], color="k", marker="o", ms=6, lw=1.6,
                                label="Centred 3-month window, 90 % interval")],
                loc="upper left", fontsize=9, frameon=False)
    _year_axis(ax_b)
    ax_b.set_title("Profile gain at S2", fontsize=11)
    # (c) the moorings, under April-July of b
    x0 = xpos(MOOR_X[0] + pd.DateOffset(years=1))
    w = xpos(MOOR_X[1] + pd.DateOffset(years=1)) - x0
    ax_c = axes_at("c", x0=x0, w=w)
    x, v, lo, hi = _monthly_gain(sq("swot", S), sq("duacs", S), S)
    _draw_ci(ax_c, x, v, lo, hi, 0.0005)
    ax_c.set_ylim(min(lo.min(), 0) - 0.002, hi.max() + 0.006)
    ax_c.set_ylabel("RMSE gain\n(kg m$^{-3}$)")
    _moorings_axis(ax_c, labels=False)
    ax_c.set_title("Profile gain, moorings", fontsize=11)
    # (d) the gradient gain by month, one row per separation
    ax_d, rng_lo, rng_hi = [], 0.0, 0.0
    for k, (x, v, lo, hi) in enumerate(_gradient_gain(S, preds["swot"]["moorings"], preds["duacs"]["moorings"])):
        ax = axes_at(f"d{k}", x0=x0, w=w, sharex=ax_c)
        _draw_ci(ax, x, v, lo, hi, 0.02)
        _moorings_axis(ax, labels=k == 3)
        rng_lo, rng_hi = min(rng_lo, lo.min()), max(rng_hi, hi.max())
        ax_d.append(ax)
    for ax in ax_d:
        ax.set_ylim(min(-0.1, rng_lo - 0.05), rng_hi + 0.3)
        ax.set_yticks([0, 0.5])
    # (e) April-July, each separation beside its row of d
    e1 = x0 - 1.05
    ax_e = [axes_at(f"d{k}", w=e1 - L) for k in range(4)]
    for ax in ax_e[1:]:
        ax.sharex(ax_e[0])
    labels = []
    for _, lo, hi in MOORING_SCALES:
        rk = mooring_pairs(S, lo, hi)[2]
        labels.append(f"{rk.min():.0f}-{rk.max():.0f} km")
    _gradient_bars(ax_e, D, labels)
    y_head = top["d0"][0]
    text_at(x0 - 0.75, (y_head + top["d3"][0] + top["d3"][1]) / 2, "Correlation, SWOT $-$ DUACS", rotation=90,
            ha="center", va="center")
    text_at((L + x0 + w) / 2, y_head - 0.55, "Correlation of the signed mixed-layer gradient, moorings", ha="center",
            va="bottom", fontsize=11)
    text_at((L + e1) / 2, y_head - 0.1, "April to July", ha="center", va="bottom", fontsize=10)
    text_at(x0 + w / 2, y_head - 0.1, "Each month", ha="center", va="bottom", fontsize=10)
    handles = [Patch(color=PRED_C, label="Prediction (SWOT)"), Patch(color=DUACS_C, label="Prediction (DUACS)"),
               Patch(color=BENCH_C["glorys12"], label=BENCH_SHORT["glorys12"]),
               Line2D([0], [0], color="k", marker="o", ms=6, lw=1.6, label="Calendar month,\n90 % interval"),
               Line2D([0], [0], color="k", marker="$*$", ms=9, ls="none", label="Interval above zero")]
    ax_d[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.06, 1.0), fontsize=9, frameon=False)
    for ch, x_in, y_in in (("a", 0.15, top["a"][0] - 0.1), ("b", 0.15, top["b"][0] - 0.1),
                           ("c", x0 - 1.15, top["c"][0] - 0.1), ("e", 0.15, y_head - 0.1),
                           ("d", x0 - 0.75, y_head - 0.1)):
        text_at(x_in, y_in, ch, fontsize=13, fontweight="bold", va="bottom")
    fig.save_bbox = None                                 # the layout is fixed in inches: no tight crop
    return fig


# ============================================================================= Figure 4
def _quads(X, Y, Zc, C, mappable):
    """Quadrilaterals between adjacent nodes of a 2-D mesh, coloured by the mean of their four corners."""
    corner = lambda A: np.stack([A[:-1, :-1], A[1:, :-1], A[1:, 1:], A[:-1, 1:]], -1)      # noqa: E731
    V = np.stack([corner(X), corner(Y), corner(Zc)], -1).reshape(-1, 4, 3)
    return V, mappable.to_rgba(corner(C).mean(-1).ravel())


def _swath_geometry(valid, G, M):
    """What the swath figures share. valid: (n_lines, n_pixels), the pixels that hold a profile; M: the mooring test
    set. k: the moorings at this overpass, north to south, and (mli, mpj) their swath pixels; (lines, pix, slat,
    slon): the section, on every swath line the pixel on the straight line through the moorings; block: the
    half-swath holding the array, trimmed until no flagged pixel is left on its rim."""
    from types import SimpleNamespace
    lat, lon = G["lat"], G["lon"]
    k = np.where((np.abs(M.time - G["time"]) < np.timedelta64(2, "h")) & (M.pass_num == G["pass_num"]))[0]
    k = k[np.argsort(-M.lat[k])]
    d2 = (lat[None] - M.lat[k, None, None]) ** 2 + ((lon[None] - M.lon[k, None, None]) * np.cos(np.deg2rad(36))) ** 2
    mli, mpj = np.unravel_index(d2.reshape(len(k), -1).argmin(1), lat.shape)
    full = valid.any(0)
    runs = np.split(np.where(full)[0], np.where(np.diff(np.where(full)[0]) > 1)[0] + 1)
    cols = max(runs, key=lambda r: np.isin(mpj, r).sum())
    lines = np.arange(lat.shape[0])
    pix = np.rint(np.polyval(np.polyfit(mli, mpj, 1), lines)).astype(int)
    on = (pix >= cols[0]) & (pix <= cols[-1])
    lines, pix = lines[on], pix[on]
    slat, slon = lat[lines, pix], lon[lines, pix]
    km_per_deg = haversine_km(slat[:-1], slon[:-1], slat[1:], slon[1:]).sum() / abs(slat[-1] - slat[0])
    r0, r1, c0, c1 = 0, lat.shape[0], cols[0], cols[-1] + 1
    while True:
        rim = {"r0": valid[r0, c0:c1], "r1": valid[r1 - 1, c0:c1], "c0": valid[r0:r1, c0], "c1": valid[r0:r1, c1 - 1]}
        worst = min(rim, key=lambda e: rim[e].mean())
        if rim[worst].all():
            break
        r0, r1, c0, c1 = r0 + (worst == "r0"), r1 - (worst == "r1"), c0 + (worst == "c0"), c1 - (worst == "c1")
    return SimpleNamespace(k=k, mli=mli, mpj=mpj, cols=cols, lines=lines, pix=pix, slat=slat, slon=slon,
                           km_per_deg=km_per_deg, block=(r0, r1, c0, c1))


def _swath_block(ax3, G, g, M, F, m_face, top, m_top, elev, azim):
    """The 3-D block: `top` on its upper face, the profiles F (n_lines, n_pixels, 40) on the faces turned to the
    viewer, the section line in red and the moorings."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    r0, r1, c0, c1 = g.block
    r, c = slice(r0, r1), slice(c0, c1)
    LO, LA, F3 = G["lon"][r, c], G["lat"][r, c], F[r, c]
    polys = [_quads(LO, LA, np.zeros_like(LO), top[r, c], m_top)]
    cam = np.array([np.cos(np.deg2rad(azim)), np.sin(np.deg2rad(azim))])
    mid = np.array([LO.mean(), LA.mean()])
    for edge in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        out = np.array([LO[edge].mean(), LA[edge].mean()]) - mid
        if out @ cam <= 0:                                           # a face turned away from the camera
            continue
        X, Y = (np.repeat(v[edge][:, None], len(Z), 1) for v in (LO, LA))
        polys.append(_quads(X, Y, np.tile(Z, (len(X), 1)), F3[edge], m_face))
    for V, fc in polys[::-1]:
        ax3.add_collection3d(Poly3DCollection(V, facecolors=fc, edgecolors=fc, linewidths=0.2, antialiased=False))
    inb = (g.lines >= r0) & (g.lines < r1) & (g.pix >= c0) & (g.pix < c1)
    ax3.plot(g.slon[inb], g.slat[inb], np.full(inb.sum(), 1.0), color="r", ls="--", lw=2, zorder=1000)
    ax3.plot(M.lon[g.k], M.lat[g.k], np.full(len(g.k), 1.5), "o", color="k", ms=7, mew=0, zorder=1001)
    ax3.set_xlim(LO.min(), LO.max())
    ax3.set_ylim(LA.min(), LA.max())
    ax3.set_zlim(Z.min(), 0)
    ax3.set_box_aspect((1.5, 1.0, 0.6), zoom=0.98)
    ax3.xaxis.set_major_locator(MultipleLocator(0.25))
    ax3.yaxis.set_major_locator(MultipleLocator(0.25))
    ax3.zaxis.set_major_locator(MultipleLocator(50))
    ax3.tick_params(labelsize=9)
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude", labelpad=8)
    ax3.text2D(-0.035, 0.66, "Depth (m)", transform=ax3.transAxes, rotation=90, va="center")
    ax3.view_init(elev=elev, azim=azim)


def swath_compare(S, G, sources, tests, elev=30, azim=-100, row_in=3.5):
    """Figure 4: the observed section along the mooring line in the first row, then one row per source: the block
    (the SSHA it was fed on top, the density anomaly on the faces) and its section along the line, with black
    contours of the anomaly every 0.1 kg/m3 and the RMSE at the moorings.
    sources: [(title, density (n, 40) at the pixels of S, SSHA grid drawn on top or None)]. Returns (fig, RMSEs)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    M = tests["moorings"]
    clim = G["clim"]
    fields = []
    for _, dens, _ in sources:
        F = np.full(G["lat"].shape + (len(DEPTH),), np.nan)
        F[S.line, S.pixel] = dens
        fields.append(F)
    g = _swath_geometry(np.isfinite(fields[0][..., 0]), G, M)
    k, o = g.k, g.k[::-1]
    vals = np.concatenate([(F[g.lines, g.pix] - clim).ravel() for F in fields] + [(M.dens[k] - clim).ravel()])
    vmax = np.ceil(np.nanpercentile(np.abs(vals), 99) * 10) / 10
    norm, cmap = Normalize(-vmax, vmax), cmocean.cm.balance
    m_face = ScalarMappable(norm, cmap)
    tops = [t for _, _, t in sources if t is not None]
    smax = np.nanpercentile(np.abs(np.concatenate([t[:, g.cols].ravel() for t in tops])), 99.5)
    m_ssh = ScalarMappable(Normalize(-smax, smax), cmocean.cm.delta)
    lev = np.linspace(-vmax, vmax, 41)
    iso = np.arange(-vmax, vmax + 0.01, 0.1)                         # the black contours of the anomaly

    n = len(sources) + 1
    head_in, foot_in = 0.5, 0.55
    H = n * row_in + head_in + foot_in
    fig = plt.figure(figsize=(13.0, H))
    W = fig.get_figwidth()
    y0 = lambda i: (foot_in + (n - 1 - i) * row_in) / H  # noqa: E731  bottom of row i
    h = row_in / H
    sec_axes, rmses = [], {}
    for j, ((title, dens, top), F) in enumerate(zip(sources, fields)):
        i = j + 1
        ax3 = fig.add_axes([0.02, y0(i) + 0.5 * h - 2.6 / H, 5.9 / W, 5.0 / H], projection="3d", facecolor="none")
        surface = (F - clim)[..., 0]
        _swath_block(ax3, G, g, M, F - clim, m_face, surface if top is None else top, m_face if top is None else m_ssh,
                     elev, azim)
        fig.text(0.035, y0(i) + (0.20 + 0.62 * 1.03) * h, chr(97 + i) + "1", fontsize=13, fontweight="bold", va="bottom")
        ax = fig.add_axes([0.56, y0(i) + 0.20 * h, 0.36, 0.62 * h], sharex=sec_axes[0] if sec_axes else None,
                          sharey=sec_axes[0] if sec_axes else None)
        sec = F[g.lines, g.pix]
        ax.contourf(g.slat, Z, (sec - clim).T, levels=lev, cmap=cmap, norm=norm, extend="both")
        ax.contour(g.slat, Z, (sec - clim).T, levels=iso, colors="k", linewidths=0.6, linestyles="solid")
        rm = np.sqrt(np.mean((F[g.mli, g.mpj] - M.dens[k]) ** 2))
        rmses[title] = float(rm)
        ax.text(0.995, 0.04, f"RMSE at the moorings {rm:.3f} kg/m³", transform=ax.transAxes, ha="right", fontsize=8,
                bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.5), zorder=20)
        ax.set_title(title, fontsize=10.5, loc="left")
        _panel_letter(ax, chr(97 + i) + "2", x=-0.13, y=1.03)
        sec_axes.append(ax)
    axT = fig.add_axes([0.56, y0(0) + 0.20 * h, 0.36, 0.62 * h], sharex=sec_axes[0], sharey=sec_axes[0])
    axT.contourf(M.lat[o], Z, (M.dens[o] - clim).T, levels=lev, cmap=cmap, norm=norm, extend="both")
    axT.contour(M.lat[o], Z, (M.dens[o] - clim).T, levels=iso, colors="k", linewidths=0.6, linestyles="solid")
    axT.set_title(f"Observed density from {len(o)} moorings", fontsize=10.5, loc="left", pad=16)
    _panel_letter(axT, "a", x=-0.13, y=1.03)
    for la, st in zip(M.lat[k], M.station[k]):
        axT.annotate(st, (la, 0), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7,
                     annotation_clip=False)
    bottom = sec_axes[-1]
    bottom.set_xlabel("Latitude along the mooring line")
    for ax in sec_axes + [axT]:
        ax.plot(M.lat[k], np.zeros(len(k)), "v", color="k", ms=6, mew=0, clip_on=False, zorder=10)
        ax.set_ylabel("Depth (m)")
        plt.setp(ax.get_xticklabels(), visible=ax is bottom)
    x0 = g.slat.min() + 4.0 / g.km_per_deg
    axT.plot([x0, x0 + 10.0 / g.km_per_deg], [-185, -185], color="k", lw=2, solid_capstyle="butt")
    axT.text(x0 + 5.0 / g.km_per_deg, -181, "10 km", ha="center", va="bottom", fontsize=8)
    sec_axes[0].set_xlim(min(g.slat.min(), M.lat[k].min() - 0.02), max(g.slat.max(), M.lat[k].max() + 0.02))
    sec_axes[0].set_ylim(*YLIM)
    # the observed row's left cell holds the colour scales
    cb = fig.colorbar(m_face, cax=fig.add_axes([0.10, y0(0) + 0.66 * h, 0.30, 0.05 * h]), orientation="horizontal",
                      extend="both")
    cb.set_label("Density anomaly from climatology (kg/m³)\nblock faces and sections; top of the product blocks")
    cb = fig.colorbar(m_ssh, cax=fig.add_axes([0.10, y0(0) + 0.24 * h, 0.30, 0.05 * h]), orientation="horizontal")
    cb.ax.xaxis.set_major_locator(MultipleLocator(0.05))
    cb.set_label("SSHA (m)\ntop of the SWOT and DUACS blocks")
    return fig, rmses


def swath_sources(S, G, model_swot, model_duacs, product="glorys12"):
    """The rows of Figure 4: the network trained and run with SWOT, its twin trained and run with DUACS, and the
    gridded product on the swath."""
    rows = [("Predicted density with SWOT input", model_swot.predict(S), G["ssh_swot"]),
            ("Predicted density with DUACS input", model_duacs.predict(S), G["ssh_duacs"])]
    if product in G:
        rows.append((f"{BENCH_SHORT[product]} density", G[product][S.line, S.pixel].astype(float), None))
    return rows


# ============================================================================= Supporting information
def _gradient_mode_content(axes, tests, tgt, layers=(("ML interior", "Mixed-layer interior"),
                                                    ("ML base", "Mixed-layer base"), ("75 m", "75 m depth"))):
    """One axis per layer: the fraction of the variance of the observed horizontal gradient between mooring pairs
    explained by the gradient of the observed anomaly truncated to mode 1, modes 1-2, 1-3 and 1-4 of the basis `tgt`,
    1 - var(g - g_k) / var(g), by separation class. Layers from the observed MLDs."""
    S = tests["moorings"]
    mm = mld(S.dens)
    M = tgt.all_modes
    truncated = [(S.anom - tgt.mean) @ M[:k].T @ M[:k] + tgt.mean for k in (1, 2, 3, 4)]
    greys, names, w = ["0.15", "0.4", "0.62", "0.82"], ["mode 1", "modes 1 - 2", "modes 1 - 3", "modes 1 - 4"], 0.19
    x = np.arange(len(MOORING_SCALES))
    ymin = 0.0
    for ax, (L, title) in zip(axes, layers):
        col = LAYERS.index(L)
        labels = []
        for j, (_, lo, hi) in enumerate(MOORING_SCALES):
            a, b, rk = mooring_pairs(S, lo, hi)
            labels.append(f"{rk.min():.0f}-{rk.max():.0f}")
            g = layer_values(S.anom, a, b, mm[a], mm[b])[:, col] / rk
            ok = np.isfinite(g)
            for i, A in enumerate(truncated):
                share = 1 - np.var(g[ok] - layer_values(A, a, b, mm[a], mm[b])[ok, col] / rk[ok]) / np.var(g[ok])
                ymin = min(ymin, share)
                ax.bar(x[j] + (i - 1.5) * w, share, w, color=greys[i], label=names[i] if j == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Mooring separation (km)")
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes:
        ax.set_ylim(min(0, np.floor(ymin * 5) / 5), 1)
    axes[0].set_ylabel("Variance of the observed horizontal\ngradient explained")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)


def pca_gradient_modes(tests, basis_anom, station="S1"):
    """Figure S1: (a-d) Figure 1d in anomaly space, the PCA basis fitted on `basis_anom`; (e-g) how much of the
    observed horizontal gradient between mooring pairs 1, 2, 3 and 4 modes can carry, in the mixed-layer interior,
    at its base and at 75 m."""
    from .models import Target
    tgt = Target(None).fit(basis_anom)
    fig = plt.figure(figsize=(10.5, 7.4), layout="constrained")
    gs = fig.add_gridspec(2, 1)
    top = gs[0].subgridspec(1, 4, wspace=0.04).subplots(sharey=True)
    bot = gs[1].subgridspec(1, 3, wspace=0.06).subplots(sharey=True)
    _recon_panels(list(top), tgt, tests, station, anomaly=True)
    _gradient_mode_content(list(bot), tests, tgt)
    for ax, letter in zip(list(top) + list(bot), "abcdefg"):
        _panel_letter(ax, letter, x=-0.03, y=1.03, ha="right")
    return fig


def s2_daily(preds, tests, bench, products=("armor3d", "glorys12", "hycom"), cols=5):
    """Figure S2: every S2 profile of the science-orbit year, one panel per pass: observed, the network with SWOT and
    with DUACS input, the climatology and the gridded products (bench = data.benchmarks()); the MLDs of the observed
    profile, of the network with SWOT input and of the climatology."""
    S = tests["s2"]
    yp_all, yd_all = preds["swot"]["s2"], preds["duacs"]["s2"]
    extra = [(p, bench[p]["s2"]) for p in products]
    o = np.argsort(S.dates.values)
    rows = int(np.ceil(len(o) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 1.83 * rows + 0.6), sharey=True, squeeze=False)
    for k, i in enumerate(o):
        ax = axes[k // cols, k % cols]
        curves = [S.dens[i], yp_all[i], yd_all[i], S.clim[i]] + [F[i] for _, F in extra]
        lo_x, hi_x = np.nanmin(curves), np.nanmax(curves)                 # each profile fills its panel
        pad = 0.06 * (hi_x - lo_x)
        ax.set_xlim(lo_x - pad, hi_x + pad)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.plot(S.dens[i], Z, color=TRUE_C, lw=2)
        ax.plot(yp_all[i], Z, color=PRED_C, lw=2)
        ax.plot(yd_all[i], Z, color=DUACS_C, lw=1.4, ls="-.", alpha=0.9)
        ax.plot(S.clim[i], Z, color=CLIM_C, ls=":", lw=2, alpha=0.7)
        for p, F in extra:
            ax.plot(F[i], Z, color=BENCH_C[p], lw=1.4, ls=BENCH_LS)
        for arr, c, ls in ((S.dens[i], "0.3", "-"), (yp_all[i], PRED_C, "-"), (S.clim[i], CLIM_C, ":")):
            z = mld(arr)[0]
            if np.isfinite(z):
                ax.axhline(-z, color=c, lw=1.0, ls=ls, alpha=0.7)
        ax.set_title(S.dates[i].strftime("%-d %B %Y"), fontsize=10)
        ax.set_ylim(*YLIM)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
    for k in range(len(o), rows * cols):
        axes[k // cols, k % cols].set_visible(False)
    for r in range(rows):
        axes[r, 0].set_ylabel("Depth (m)", fontsize=10)
    for c in range(cols):
        last = max(r for r in range(rows) if axes[r, c].get_visible())
        axes[last, c].set_xlabel("Density (kg/m³)", fontsize=10)
    handles = LEGEND + [Line2D([0], [0], color=BENCH_C[p], lw=1.4, ls=BENCH_LS, label=BENCH_SHORT[p]) for p in products]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.0), framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _band(ax, arr, color, alpha):
    """Mean of the rows of arr against depth with a +/-1 SD band (clipped at zero); returns the mean."""
    mu, sd = np.nanmean(arr, 0), np.nanstd(arr, 0)
    ax.fill_betweenx(Z, np.clip(mu - sd, 0, None), mu + sd, color=color, alpha=alpha, lw=0)
    return mu


def gradients_by_scale(preds, tests, bench, product="glorys12"):
    """Figure S3: the horizontal density gradient between mooring pairs by separation class (columns), April-July
    2023. (a-c) Mean |d rho/dx| against depth with +/-1 SD bands across pairs, observed against the network with SWOT
    input, with DUACS input, and the product (bench = data.benchmarks()). (d, e) The distribution (log scale) of the
    signed gradient, north minus south, in the mixed-layer interior and at its base, on the pairs every estimate
    covers."""
    S = tests["moorings"]
    rows = [("Network, SWOT input", preds["swot"]["moorings"], PRED_C, "-"),
            ("Same network, DUACS input", preds["duacs"]["moorings"], DUACS_C, "-."),
            (BENCH_SHORT[product], bench[product]["moorings"], BENCH_C[product], BENCH_LS)]
    pdf_layers = [("ML interior", "Mixed-layer\ninterior"), ("ML base", "Mixed-layer\nbase")]
    mm = mld(S.dens)
    covered = np.isfinite(S.dens).all(1) & np.all([np.isfinite(Y).all(1) for _, Y, _, _ in rows], axis=0)
    fig, axes = plt.subplots(len(rows) + len(pdf_layers), len(MOORING_SCALES), figsize=(10.5, 13.4),
                             gridspec_kw=dict(height_ratios=[1] * len(rows) + [1.1] * len(pdf_layers)))
    for j, (_, lo, hi) in enumerate(MOORING_SCALES):
        a, b, rk = mooring_pairs(S, lo, hi)
        north = S.lat[a] >= S.lat[b]
        a, b = np.where(north, a, b), np.where(north, b, a)
        gt = np.abs(S.dens[a] - S.dens[b]) / rk[:, None] * 1000
        xmax = 1.05 * np.nanmax(np.nanmean(gt, 0) + np.nanstd(gt, 0))          # the observed band fills the panel
        for i, (_, Y, c, _) in enumerate(rows):
            ax = axes[i, j]
            mt = _band(ax, gt, TRUE_C, 0.15)
            mp = _band(ax, np.abs(Y[a] - Y[b]) / rk[:, None] * 1000, c, 0.30)
            ax.plot(mt, Z, color=TRUE_C, lw=1.8)
            ax.plot(mp, Z, color=c, lw=1.8)
            ax.set_xlim(0, xmax)
            ax.set_ylim(*YLIM)
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(f"Pairs {rk.min():.0f}-{rk.max():.0f} km")
            if i == len(rows) - 1:
                ax.set_xlabel(r"$|\partial\rho/\partial x|$  (g m$^{-3}$ km$^{-1}$)")
            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
        sel = covered[a] & covered[b]
        a, b, rk = a[sel], b[sel], rk[sel]
        for q, (L, _) in enumerate(pdf_layers):
            ax = axes[len(rows) + q, j]
            col = LAYERS.index(L)
            g_obs = layer_values(S.dens, a, b, mm[a], mm[b])[:, col] / rk * 1000
            fin = np.isfinite(g_obs)
            lim = np.percentile(np.abs(g_obs[fin]), 99.5)
            bins = np.linspace(-lim, lim, 41)
            cen = 0.5 * (bins[1:] + bins[:-1])
            ax.plot(cen, np.histogram(g_obs[fin], bins, density=True)[0], color=TRUE_C, lw=2.0)
            for _, Y, c, ls in rows:
                g = layer_values(Y, a, b, mm[a], mm[b])[:, col] / rk * 1000
                ok = fin & np.isfinite(g)
                ax.plot(cen, np.histogram(g[ok], bins, density=True)[0], color=c, lw=1.6, ls=ls)
            ax.set_yscale("log")
            ax.set_xlim(-lim, lim)
            ax.grid(alpha=0.25)
            ax.set_xlabel(r"$\partial\rho/\partial x$  (g m$^{-3}$ km$^{-1}$)")
            if j == 0:
                ax.set_ylabel("Probability density")
    for i in range(len(rows)):
        axes[i, 0].set_ylabel("Depth (m)")
    h = [Line2D([0], [0], color=TRUE_C, lw=1.8, label="Observed")] + \
        [Line2D([0], [0], color=c, lw=1.8, ls=ls, label=lab) for lab, _, c, ls in rows]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=10, bbox_to_anchor=(0.5, -0.005), framealpha=0.9)
    fig.tight_layout(rect=[0.05, 0.045, 1, 1], h_pad=1.2)
    margin = [lab.replace(", ", "\n") for lab, _, _, _ in rows] + [lab for _, lab in pdf_layers]
    for i, lab in enumerate(margin):                                 # row labels in the margin, after the layout
        bb = axes[i, 0].get_position()
        fig.text(0.012, 0.5 * (bb.y0 + bb.y1), lab, rotation=90, va="center", ha="left", fontweight="bold")
        fig.text(0.012, bb.y1 + 0.008, "abcde"[i], fontsize=13, fontweight="bold", va="bottom", ha="left")
    return fig


def _glider_transects(fig, preds, tests, bench, product, n_cols=2):
    """The n_cols longest glider transects (loiters excluded), in time order, one column each: the density anomaly
    from the climatology with isopycnals every 0.25 kg/m3, observed, with SWOT and with DUACS input, and the product;
    the RMSE over each reconstruction. Returns the axes (rows x transects)."""
    S = tests["gliders"]
    loi = loiter_mask(S.glider, S.lat, S.lon, S.dates)
    t = S.dates.values
    cands = []
    for seg in segments(S.glider, S.lat, S.lon, S.dates):
        seg = seg[~loi[seg]]
        if len(seg) < 8:
            continue
        o = seg[np.argsort(t[seg])]
        x = np.concatenate([[0.0], np.cumsum(haversine_km(S.lat[o][:-1], S.lon[o][:-1], S.lat[o][1:], S.lon[o][1:]))])
        cands.append((x[-1], o, x))
    secs = sorted(sorted(cands, key=lambda c: -c[0])[:n_cols], key=lambda c: t[c[1][0]])
    fields = [("Observed", S.dens), ("Prediction (SWOT)", preds["swot"]["gliders"]),
              ("Prediction (DUACS)", preds["duacs"]["gliders"]), (BENCH_SHORT[product], bench[product]["gliders"])]
    vmax = np.ceil(np.nanpercentile(np.abs(np.concatenate([S.anom[o].ravel() for _, o, _ in secs])), 99) * 10) / 10
    iso = np.arange(1024.0, 1027.5, 0.25)
    axes = fig.subplots(len(fields), len(secs), sharey=True, sharex="col", squeeze=False)
    for c, (_, o, x) in enumerate(secs):
        obs = S.dens[o].T
        for r, (lab, F) in enumerate(fields):
            ax = axes[r, c]
            R = F[o].T
            pc = ax.pcolormesh(x, DEPTH, R - S.clim[o].T, cmap=cmocean.cm.balance, shading="nearest", vmin=-vmax,
                               vmax=vmax)
            ax.clabel(ax.contour(x, DEPTH, R, levels=iso, colors="k", linewidths=0.7), fmt="%.2f", fontsize=6)
            if r == 0:
                ax.set_title(f"{S.glider[o][0]}: {pd.Timestamp(t[o][0]):%b %d} - {pd.Timestamp(t[o][-1]):%b %d %Y}"
                             f"\n{len(o)} profiles, {x[-1]:.0f} km", fontsize=10)
            else:
                ok = np.isfinite(R).all(0) & np.isfinite(obs).all(0)
                rm = np.sqrt(np.mean((R[:, ok] - obs[:, ok]) ** 2))
                ax.set_title(f"RMSE {rm:.3f} kg/m³" + (f" ({ok.sum()}/{len(o)} profiles)" if ok.sum() < len(o) else ""),
                             fontsize=10)
            if r == len(fields) - 1:
                ax.set_xlabel("Along-track distance (km)")
            if c == 0:
                ax.set_ylabel(f"{lab}\nDepth (m)")
    axes[0, 0].set_ylim(200, 0)
    fig.colorbar(pc, ax=axes, label="Density anomaly from climatology (kg/m³)", aspect=60, pad=0.01)
    return axes


def gliders(preds, tests, D, bench, product="glorys12"):
    """Figure S4: (a-d) the two longest glider transects, observed and reconstructed (bench = data.benchmarks());
    (e) the correlation of the observed and predicted signed gradient in the mixed-layer interior between glider
    pairs, by separation, with 90% intervals (D = data.gradients())."""
    fig = plt.figure(figsize=(10.5, 2.4 * 4 + 3.6), constrained_layout=True)
    top, bot = fig.subfigures(2, 1, height_ratios=[2.4 * 4, 3.1], hspace=0.05)
    axes = _glider_transects(top, preds, tests, bench, product)
    for r in range(axes.shape[0]):
        _panel_letter(axes[r, 0], chr(97 + r), x=-0.3, y=1.02)
    ax = bot.subplots(1, 1)
    scales = [s for s, *_ in GLIDER_SCALES]
    ax.axhline(0, color="0.5", lw=0.8)
    ax.grid(alpha=0.3, axis="y")
    series = [("network", "SWOT", PRED_C, "o"), ("network", "DUACS", DUACS_C, "s"),
              (product, "analysis", BENCH_C[product], "^")]
    for k, (model, field, c, mk) in enumerate(series):
        q = D[(D.platform == "gliders") & (D.layer == "ML interior") & (D.model == model) & (D.field == field)]
        q = q.set_index("scale").reindex(scales)
        ax.errorbar(np.arange(len(scales)) + 0.2 * (k - 1), q.r, yerr=[q.r - q.r_lo, q.r_hi - q.r], fmt=mk, color=c,
                    ms=7, mew=1.6, capsize=2, lw=1.2)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(scales)
    ax.set_xlabel("Separation of the glider pairs")
    ax.set_ylim(-0.25, 0.85)
    ax.set_ylabel("Correlation of signed gradient,\nmixed-layer interior")
    _panel_letter(ax, chr(97 + axes.shape[0]), x=-0.12, y=1.02)
    h = [Line2D([], [], marker="o", ls="", color=PRED_C, label="SWOT input"),
         Line2D([], [], marker="s", ls="", color=DUACS_C, label="DUACS input"),
         Line2D([], [], marker="^", ls="", color=BENCH_C[product], label=BENCH_SHORT[product]),
         Line2D([], [], color="0.3", lw=1.2, label="90% CI")]
    bot.legend(handles=h, loc="outside lower center", ncol=len(h), fontsize=9)
    return fig


LAGS_H = np.arange(-48, 49)                            # Figure S5a: hours from the pass to the observed profile
M2_H = 12.42                                           # the semidiurnal tide's period (h)
GLIDER_BINS = ((0, 1.5), (1.5, 3), (3, 4.7), (4.7, 7.7), (7.7, 9.3), (9.3, 12.1))    # |profile - pass| (h), S5b
ARGO_BINS = ((0, 3), (3, 6), (6, 12), (12, 18), (18, 24))


def _lag_gain(S, pred_swot, pred_duacs, records):
    """The RMSE gain with the predictions of the pass against the observed profile at pass time + lag, for each lag
    in LAGS_H: the hourly profile nearest to it within 40 min (records: {station: hourly profiles as a DataFrame})."""
    stations = S.station if "station" in S else np.full(S.n, "S2")
    out = []
    for lag in LAGS_H:
        t = S.dates + pd.Timedelta(hours=int(lag))
        y = np.full(S.dens.shape, np.nan)
        for st, R in records.items():
            m = stations == st
            y[m] = R.reindex(t[m], method="nearest", tolerance=pd.Timedelta("40min")).to_numpy()
        ok = np.isfinite(y).all(1)
        out.append(_gain(((pred_swot - y)[ok] ** 2).mean(1), ((pred_duacs - y)[ok] ** 2).mean(1)))
    return np.array(out)


def _binned_gain(es, ed, dt, bins, blocks, n=2000):
    """The RMSE gain in bins of |profile time - pass time| (h), with 90 % intervals from n block-bootstrap resamples.
    Returns (bin centres, value, lo, hi)."""
    picks = boot_picks(blocks, n)
    out = []
    for lo, hi in bins:
        m = (dt >= lo) & (dt < hi)
        bs = [_gain(es[q], ed[q]) if len(q) > 3 else np.nan for q in (p[m[p]] for p in picks)]
        out.append(((lo + hi) / 2, _gain(es[m], ed[m]), *np.nanpercentile(bs, [5, 95])))
    return (np.array(c) for c in zip(*out))


def tide_lag(preds, tests, pairs, oof):
    """Figure S5: SWOT's profile gain, RMSE with DUACS input minus with SWOT input over 0-200 m, against the time
    between the observed profile and the pass. (a) Mooring array and S2: the predictions stay those of the pass and
    the observed hourly profile is taken up to 48 h before and after it; dotted lines every M2 period. (b) Gliders
    and Argo pairs (oof: the out-of-fold predictions at the pairs with SWOT and with DUACS input), binned by the time
    between the profile and the pass, with 90 % intervals from resampling days (gliders) and floats (pairs)."""
    S, S2 = tests["moorings"], tests["s2"]
    gray = "0.45"
    moorings = {st: pd.DataFrame(dens, index=pd.DatetimeIndex(t)) for st, (t, dens) in data.moorings_hourly().items()}
    H = data.s2_hourly()
    s2 = pd.DataFrame(H["dens"].astype(float), index=pd.DatetimeIndex(H["time"]))
    s2 = {"S2": s2[~s2.index.duplicated()].sort_index()}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    a = ax[0]
    a.plot(LAGS_H, _lag_gain(S, preds["swot"]["moorings"], preds["duacs"]["moorings"], moorings), color="k", lw=2,
           label="Mooring array")
    a.plot(LAGS_H, _lag_gain(S2, preds["swot"]["s2"], preds["duacs"]["s2"], s2), color=gray, lw=1.5, ls="-.",
           label="S2")
    for k in range(-3, 4):
        a.axvline(k * M2_H, color=gray, lw=0.7, ls=":")
    a.axhline(0, color=gray, lw=1)
    a.set_xlim(-48, 48)
    a.set_xticks(np.arange(-48, 49, 12))
    a.set_xlabel("Time of the observed profile minus time of the pass (h)")
    a.set_ylabel("RMSE gain (kg m$^{-3}$)")
    a.set_title("(a) Mooring array and S2", fontsize=11)
    a.legend(fontsize=10, loc="lower left")
    a = ax[1]
    G = tests["gliders"]
    sq = lambda pred, T: ((pred - T.dens) ** 2).mean(1)                      # noqa: E731
    series = [(_binned_gain(sq(preds["swot"]["gliders"], G), sq(preds["duacs"]["gliders"], G), np.abs(G.dt_h),
                            GLIDER_BINS, data.bootstrap_blocks(G)), "o-", "k", f"Gliders, {G.n} profiles"),
              (_binned_gain(sq(oof[0], pairs), sq(oof[1], pairs), np.abs(pairs.dt_h), ARGO_BINS,
                            pd.factorize(pairs.wmo)[0]), "s--", gray, f"Argo pairs, {pairs.n} profiles")]
    for (x, y, lo, hi), fmt, c, lab in series:
        a.errorbar(x, y, yerr=[y - lo, hi - y], fmt=fmt, color=c, lw=1.8, ms=5, capsize=3, label=lab)
    for k in (1, 2):
        a.axvline(k * M2_H, color=gray, lw=0.7, ls=":")
    a.axhline(0, color=gray, lw=1)
    a.set_xlim(0, 25)
    a.set_xticks(np.arange(0, 25, 3))
    a.set_xlabel("Time between the profile and the pass (h)")
    a.set_ylabel("RMSE gain (kg m$^{-3}$)")
    a.set_title("(b) Gliders and Argo pairs, 90% intervals", fontsize=11)
    a.legend(fontsize=10, loc="lower right")
    for a in ax:
        a.grid(alpha=0.3)
    return fig
