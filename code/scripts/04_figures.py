"""The figures of the paper and of its supporting information, into figures/.

    python scripts/04_figures.py [fig1 fig2 fig3 fig4 figS1 figS2 figS3 figS4 figS5]     # all by default

The notebooks in notebooks/ draw the same figures with the numbers the text quotes about each.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")

from swot2argo import data, models, plots                             # noqa: E402
from swot2argo.config import MODELS_DIR                              # noqa: E402

MAIN_SWATH = "swath_20230508_13.npz"        # the 8 May 2023 overpass (Cal/Val pass 26): Figures 1b and 4
OOF = "pretrained|ssh,sst,mdt|4||"          # the selected network in results/cv_predictions.npz, + 'swot' or 'duacs'
NAMES = ["fig1", "fig2", "fig3", "fig4", "figS1", "figS2", "figS3", "figS4", "figS5"]


def fig4():
    T = data.tests()
    G = data.swath(MAIN_SWATH)
    S = data.swath_pixels(G)
    m_swot = models.load(os.path.join(MODELS_DIR, "headline_swot.pt"))
    m_duacs = models.load(os.path.join(MODELS_DIR, "headline_duacs.pt"))
    fig, rmses = plots.swath_compare(S, G, plots.swath_sources(S, G, m_swot, m_duacs), T)
    for k, v in rmses.items():
        print(f"  {k}: RMSE along the mooring line {v:.3f} kg/m3")
    return fig


def figS5():
    cv = data.cv_predictions()
    return plots.tide_lag(data.predictions(), data.tests(), data.pairs(), (cv[OOF + "swot"], cv[OOF + "duacs"]))


def main(only):
    T = data.tests()
    FIGURES = {
        "fig1": lambda: plots.overview(T, data.history().anom, MAIN_SWATH),
        "fig2": lambda: plots.disagreement_examples(data.predictions(), T),
        "fig3": lambda: plots.season_gain(data.predictions(), T, data.gradients(), data.swath_crossovers()),
        "fig4": fig4,
        "figS1": lambda: plots.pca_gradient_modes(T, data.history().anom),
        "figS2": lambda: plots.s2_daily(data.predictions(), T, data.benchmarks()),
        "figS3": lambda: plots.gradients_by_scale(data.predictions(), T, data.benchmarks()),
        "figS4": lambda: plots.gliders(data.predictions(), T, data.gradients(), data.benchmarks()),
        "figS5": figS5,
    }
    for name, make in FIGURES.items():
        if only and name not in only:
            continue
        plots.save(make(), name + ".png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", help=f"{', '.join(NAMES)} (default: all)")
    args = ap.parse_args()
    unknown = set(args.figures) - set(NAMES)
    if unknown:
        ap.error(f"unknown figure(s) {sorted(unknown)}: choose from {', '.join(NAMES)}")
    main(args.figures)
