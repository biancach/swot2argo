"""Score the models at the zero-shot test sets and write the numeric tables of the Supporting Information.

    python scripts/03_evaluate.py          # a few minutes (block bootstraps)

Reads results/predictions.npz and configurations.npz (scripts/02_train.py), results/cv.csv (scripts/01_select.py)
and the data folder. Writes
  results/test_scores.csv          RMSE 0-200 m of every estimate at each test set, its 90% interval, the RMSE by
                                   depth band, and the difference from the climatology with its interval
  results/gradients.csv            horizontal-gradient skill at the moorings and gliders, every layer and separation
  results/tables/table_s3.csv ...  SI Tables S3 to S8 (Tables S1 and S2 describe the data and the settings:
  table_s8.csv                     data/README.md, swot2argo/config.py and models.py)
  results/summary.txt              the tables as text
Sites in the SI order: moorings, S2, gliders. Every interval is a 90% block bootstrap (blocks: station x ISO week at
the moorings, ISO week at S2, day at the gliders).
"""
import os

import numpy as np
import pandas as pd

from swot2argo import data, evaluate, gradients, models
from swot2argo.config import EXCLUDED_INPUTS, FEATURES, MODELS_DIR, RESULTS_DIR

SITES = ("moorings", "s2", "gliders")
TABLES = os.path.join(RESULTS_DIR, "tables")
PRODUCTS = {"glorys12": "GLORYS12", "armor3d": "ARMOR3D", "hycom": "HYCOM", "glo12": "GLO12", "pwp": "PWP"}
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def star(sig):
    return "*" if sig else " "


def write(df, name, title):
    df.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
    say(f"\n{title}")


# ----------------------------------------------------------------------------- Table S3: profile RMSE
def table_s3(T, P, B, mean_anomaly):
    est = {"Climatology": {t: T[t].clim for t in SITES},
           "Climatology + mean Argo anomaly": {t: T[t].clim + mean_anomaly for t in SITES},
           "Network, SWOT input": P["swot"], "Network, DUACS input": P["duacs"]}
    hourly = data.s2_hourly()
    for lag in (3, 7):
        est[f"Persistence, in situ profile {lag} days old"] = {
            "moorings": evaluate.persistence(T["moorings"], lag), "s2": evaluate.persistence(T["s2"], lag, hourly)}
    est.update({lab: B[p] for p, lab in PRODUCTS.items()})
    rows = []
    for name, E in est.items():
        for t in SITES:
            if t not in E:
                continue
            S, x = T[t], E[t]
            ok = np.isfinite(x).all(1)
            r = dict(estimate=name, site=t, n=int(ok.sum()), rmse=evaluate.rmse(x[ok], S.dens[ok]))
            if not name.startswith(("Network, SWOT", "Persistence")):      # network with SWOT input minus this row
                d = evaluate.bootstrap_vs(P["swot"][t], x, S)
                r.update(network_minus=d["diff"], lo=d["lo"], hi=d["hi"], differs=d["significant"])
            rows.append(r)
    D = pd.DataFrame(rows)
    write(D, "table_s3", "Table S3. Total density RMSE 0-200 m (kg/m3); * = differs from the network with SWOT input "
          "(paired 90% interval excludes 0); persistence on the samples with a profile that many days earlier")
    say(f"{'estimate':40s}" + "".join(f"{t:>16s}" for t in SITES))
    for name, g in D.assign(mark=D.get("differs", False).eq(True).map(star)).groupby("estimate", sort=False):
        g = g.set_index("site")
        say(f"{name:40s}" + "".join(f"{g.rmse[t]:9.3f}{g.mark[t]} n={g.n[t]:<4d}" if t in g.index else f"{'--':>16s}"
                                    for t in SITES))
    for lag in range(1, 8):                                                # Text S4: how old may the profile be?
        cells = []
        for t in ("moorings", "s2"):
            x = evaluate.persistence(T[t], lag, hourly if t == "s2" else None)
            ok = np.isfinite(x).all(1)
            cells.append(f"{t} {evaluate.rmse(x[ok], T[t].dens[ok]):.3f} (network {evaluate.rmse(P['swot'][t][ok], T[t].dens[ok]):.3f})")
        say(f"  persistence {lag} d: " + ", ".join(cells))


# ----------------------------------------------------------------------------- Table S4: per-mode skill
def table_s4(cv, pairs, T, P, basis):
    rows = []
    sets = [("ssh",), ("sst",), ("mdt",), ("ssh", "sst"), ("ssh", "mdt"), ("sst", "mdt"), ("ssh", "sst", "mdt")]
    for fs in sets:
        r = cv.set_index("key").loc[evaluate.candidate_key("pretrained", fs, 4)]
        rows.append(dict(part="cross-validation", inputs=models.label(fs), metric="R2", cv_rmse=r.rmse,
                         **{f"mode{j}": r[f"r2_mode{j}"] for j in range(1, 5)}))
    sd = basis.encode(pairs.anom).std(0)
    rows.append(dict(part="cross-validation", inputs="SD of the observed coefficient", metric="SD",
                     **{f"mode{j + 1}": sd[j] for j in range(4)}))
    for t in SITES:
        s = evaluate.mode_skill(P["swot"][t], T[t], basis)
        for metric in ("r2", "skill"):
            rows.append(dict(part="test sets", inputs=f"{t} (n={T[t].n})", metric=metric,
                             **{f"mode{j + 1}": s[metric][j] for j in range(4)}))
    D = pd.DataFrame(rows)
    write(D, "table_s4", "Table S4. Per-mode skill on the four modes of the pretraining basis: cross-validation of the "
          "pretrained network over the pairs (SWOT input), and the network with SWOT input at the test sets")
    for _, r in D.iterrows():
        cvr = f"{r.cv_rmse:.3f}" if np.isfinite(r.get("cv_rmse", np.nan)) else ""
        say(f"  {r.inputs:32s} {r.metric:6s} {cvr:>6s} " + " ".join(f"{r[f'mode{j}']:+6.2f}" for j in range(1, 5)))


# ----------------------------------------------------------------------------- Table S5: SWOT against DUACS input
def table_s5(T, P):
    D = pd.concat([evaluate.swot_vs_duacs(P, T).assign(bin="all samples"), evaluate.disagreement_bins(P, T)])
    D["site"] = pd.Categorical(D.pop("test"), SITES)
    D = D.sort_values("site", kind="stable")
    write(D, "table_s5", "Table S5. Network with SWOT input minus with DUACS input, RMSE 0-200 m (kg/m3), all samples "
          "and binned by |SWOT - DUACS SSHA| at the sample (thresholds: its median, 75th and 90th percentile)")
    for _, r in D.iterrows():
        cut = f"from {r.dssh_cm_from:3.1f} cm" if np.isfinite(r.get("dssh_cm_from", np.nan)) else ""
        n = int(r.n) if np.isfinite(r.get("n", np.nan)) else T[r.site].n
        say(f"  {r.site:9s} {r['bin']:12s} {cut:12s} n={n:5d}  SWOT {r.swot:.3f}  DUACS {r.duacs:.3f}  "
            f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]{star(r.significant)}")


# ----------------------------------------------------------------------------- Table S6: gradients
LABEL_SEP = {"5-15 km": "9-12 km", "15-25 km": "18-22 km", "25-35 km": "29-31 km", "35-60 km": "39-60 km"}


def table_s6(G):
    rows = []
    for layer in ("ML interior", "75 m", "ML base"):
        for platform in ("moorings", "gliders"):
            g = G[(G.platform == platform) & (G.layer == layer)]
            for scale in g.scale.unique():
                f = g[g.scale == scale].set_index(["model", "field"])
                s, d = f.loc[("network", "SWOT")], f.loc[("network", "DUACS")]
                rows.append(dict(layer=layer, platform=platform, separation=LABEL_SEP.get(scale, scale), n=int(s.n),
                                 obs_amplitude=s.obs_rms, r_swot=s.r, r_swot_ci_has_0=s.r_lo <= 0 <= s.r_hi,
                                 r_duacs=d.r, r_duacs_ci_has_0=d.r_lo <= 0 <= d.r_hi, dr=s.dr,
                                 dr_differs=not s.dr_lo <= 0 <= s.dr_hi, amp_swot=s.amp, amp_duacs=d.amp,
                                 sign_swot=s.sign, r_glorys12=f.loc[("glorys12", "analysis")].r,
                                 r_armor3d=f.loc[("armor3d", "analysis")].r))
    D = pd.DataFrame(rows)
    write(D, "table_s6", "Table S6. Signed horizontal gradient between profile pairs: observed rms (g/m3/km), "
          "correlation r with the observed one († = 90% interval includes 0), dr = r SWOT - r DUACS (* = interval "
          "excludes 0), amplitude, fraction of signs right, and r of GLORYS12 and ARMOR3D")
    for _, r in D.iterrows():
        say(f"  {r.layer:11s} {r.platform:8s} {r.separation:8s} n={r.n:4d} obs {r.obs_amplitude:5.1f}  "
            f"r {r.r_swot:+.2f}{'†' if r.r_swot_ci_has_0 else ' '} {r.r_duacs:+.2f}{'†' if r.r_duacs_ci_has_0 else ' '} "
            f"dr {r.dr:+.2f}{star(r.dr_differs)}  amp {r.amp_swot:.2f} {r.amp_duacs:.2f}  sign {r.sign_swot:.2f}  "
            f"GLORYS12 {r.r_glorys12:+.2f}  ARMOR3D {r.r_armor3d:+.2f}")


# ----------------------------------------------------------------------------- Table S7: mixed-layer depth
def table_s7(T, P, B, scores):
    rows = []
    for t in SITES:
        for name, key, x in [("Climatology", "climatology", T[t].clim), ("Network, SWOT input", "network | SWOT", P["swot"][t]),
                             ("Network, DUACS input", "network | DUACS", P["duacs"][t])] + [
                (PRODUCTS[p], f"{p} | analysis", B[p][t]) for p in ("glorys12", "armor3d", "hycom")]:
            sc = scores.set_index("model").loc[key]
            rows.append(dict(site=t, model=name, n=int(np.isfinite(x).all(1).sum()), **evaluate.mld_skill(x, T[t]),
                             rmse_0_30=sc[f"{t} 0-30 m"], rmse_30_100=sc[f"{t} 30-100 m"]))
    D = pd.DataFrame(rows)
    write(D, "table_s7", "Table S7. Mixed-layer depth (0.125 kg/m3 above 10 m): MAE, bias (predicted - observed) and r "
          "(m); total density RMSE over 0-30 m and 30-100 m (kg/m3)")
    for _, r in D.iterrows():
        say(f"  {r.site:9s} {r.model:22s} n={r.n:5d}  MAE {r.mae:4.1f}  bias {r.bias:+5.1f}  r {r.r:.2f}  "
            f"RMSE 0-30 m {r.rmse_0_30:.3f}  30-100 m {r.rmse_30_100:.3f}")


# ----------------------------------------------------------------------------- Table S8: the configurations
def table_s8(cv, T, P):
    C = np.load(os.path.join(RESULTS_DIR, "configurations.npz"))
    names = list(dict.fromkeys(k.split("#")[0] for k in C.files))         # the order of Table S8
    head = evaluate.candidate_key("pretrained", FEATURES, 4)
    names.insert(names.index(evaluate.candidate_key("pretrained", ("ssh", "mdt"), 4)) + 1, head)
    picked = {p for _, p in evaluate.selection(cv).values()}
    cvk = cv.set_index("key")
    rows = []
    for k in names:
        fam, fs, nm = k.split("|")[:3]
        fs = tuple(fs.split(","))
        note = ["selected"] * (k in picked) + ["reported"] * (k == head) + ["excluded"] * bool(set(fs) & set(EXCLUDED_INPUTS))
        pred = P["swot"] if k == head else {t: C[f"{k}#{t}"] for t in SITES}
        rows.append(dict(family=fam, inputs=models.label(fs), output="40 levels" if nm == "levels" else f"{nm} modes",
                         note=", ".join(note), cv=cvk.rmse[k], **{t: evaluate.rmse(pred[t], T[t].dens) for t in SITES}))
    D = pd.DataFrame(rows)
    write(D, "table_s8", "Table S8. The configurations tried, SWOT input: cross-validation RMSE over the pairs, and RMSE "
          "0-200 m at the test sets after training on all pairs (kg/m3)")
    for _, r in D.iterrows():
        lab = f"{r.family}, {r.inputs}, {r.output}" + (f" ({r.note})" if r.note else "")
        say(f"  {lab:75s} CV {r.cv:.3f}  " + "  ".join(f"{t} {r[t]:.3f}" for t in SITES))


def main():
    os.makedirs(TABLES, exist_ok=True)
    T, P, B, pairs = data.tests(), data.predictions(), data.benchmarks(), data.pairs()
    mean_anomaly = data.history().anom.mean(0)                    # the mean anomaly of the pretraining profiles
    estimates = {"network | SWOT": P["swot"], "network | DUACS": P["duacs"], "climatology": {t: T[t].clim for t in T},
                 "climatology + mean Argo anomaly": {t: T[t].clim + mean_anomaly for t in T}}
    estimates.update({f"{b} | analysis": B[b] for b in B})
    scores = evaluate.score_table(estimates, T)
    scores.to_csv(os.path.join(RESULTS_DIR, "test_scores.csv"), index=False)

    G = []
    for t in ("moorings", "gliders"):
        g = gradients.gradient_scores(T[t], {"SWOT": P["swot"][t], "DUACS": P["duacs"][t]})
        G.append(g.assign(model="network"))
        for b in B:
            G.append(gradients.gradient_scores(T[t], {"analysis": B[b][t]}, compare=()).assign(model=b))
    G = pd.concat(G)
    G.insert(0, "model", G.pop("model"))
    G.to_csv(os.path.join(RESULTS_DIR, "gradients.csv"), index=False)

    cv = pd.read_csv(os.path.join(RESULTS_DIR, "cv.csv"), dtype={"n_modes": str})
    basis = models.load(os.path.join(MODELS_DIR, "headline_swot.pt")).target
    table_s3(T, P, B, mean_anomaly)
    table_s4(cv, pairs, T, P, basis)
    table_s5(T, P)
    table_s6(G)
    table_s7(T, P, B, scores)
    table_s8(cv, T, P)
    open(os.path.join(RESULTS_DIR, "summary.txt"), "w").write("\n".join(lines).lstrip() + "\n")
    print("\nwrote results/test_scores.csv, gradients.csv, tables/table_s3-s8.csv, summary.txt")


if __name__ == "__main__":
    main()
