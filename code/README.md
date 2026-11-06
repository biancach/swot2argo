# swot2argo

Upper-ocean density profiles (0-200 m) reconstructed from SWOT sea surface height with a neural network trained on
Argo, tested at the SWOT Cal/Val moorings, two gliders and a mooring at the S2 site in the California Current System.

Code for *Reconstructing 3D upper-ocean density from SWOT's wide-swath sea surface height using machine learning*
(Champenois, Ali and Bodner). The data folder is described in [../data/README.md](../data/README.md).

## Quick look

The notebooks draw the figures of the paper, with the numbers the text quotes about each, from the shipped results
(no training needed):

| notebook | figure |
|---|---|
| [notebooks/fig1_data_and_basis.ipynb](notebooks/fig1_data_and_basis.ipynb) | the training and test data, the hourly S2 record, the PCA basis |
| [notebooks/fig2_where_the_inputs_differ.ipynb](notebooks/fig2_where_the_inputs_differ.ipynb) | overpasses on which SWOT and DUACS differ at the site |
| [notebooks/fig3_season_of_the_gain.ipynb](notebooks/fig3_season_of_the_gain.ipynb) | when SWOT input beats DUACS input: the swath, S2, the moorings, the horizontal gradients |
| [notebooks/fig4_swath.ipynb](notebooks/fig4_swath.ipynb) | one overpass reconstructed in 3-D, the section along the mooring line |
| [notebooks/supporting_figures.ipynb](notebooks/supporting_figures.ipynb) | Figures S1-S5 of the Supporting Information |

## Setup

```
conda env create -f environment.yml && conda activate swot2argo   # or: pip install -r requirements.txt
pip install -e .
```

The data are in [../data/](../data/), next to this folder, which is where the code looks for them (it also accepts a
`data/` folder inside this one, or the path in the environment variable `SWOT2ARGO_DATA`). A GPU is only needed to
retrain.

## Running

```
python scripts/01_select.py      # cross-validation of the candidate models over the SWOT-Argo pairs (about an hour on a GPU)
python scripts/02_train.py       # train the selected network and the other configurations of Table S8 (about 30 min)
python scripts/03_evaluate.py    # scores at the test sets and Tables S3-S8 (a few minutes)
python scripts/04_figures.py     # Figures 1-4 and S1-S5 (or name some: fig3 figS5)
```

`results/` ships with the output of every step (the trained network, its predictions, the cross-validation table,
the tables), so each step and the notebooks run without the ones before it. A step rewrites what it produces;
`python scripts/02_train.py --members 2` is a quick smoke test.

The shipped network (`results/models/`) and the predictions of the configurations of Table S8 are those of the paper.
Training on a GPU is not repeatable bit for bit, so retraining gives RMSEs within about 0.0003 kg/m3 of them, and
the shipped cross-validation (`results/cv.csv`, a run of `01_select.py`) can differ from the Supporting Information
in the last digit of a few entries of Tables S4 and S8.

| in the paper | made by | written to |
|---|---|---|
| Figures 1-4, S1-S5 | `04_figures.py` | `figures/` |
| Tables S3-S7, the numbers of the text | `03_evaluate.py` | `results/tables/`, `results/summary.txt` |
| Table S8, cross-validation part of Table S4 | `01_select.py`, then `02_train.py` and `03_evaluate.py` | `results/cv.csv`, `results/tables/` |
| Tables S1, S2 (data sets, training settings) | (descriptive) | `../data/README.md`, `swot2argo/config.py`, `swot2argo/models.py` |

The model is chosen by cross-validation over the 424 SWOT-Argo pairs only (SI Text S3): `01_select.py` prints the
pick of the one-standard-error rule in each family, and the pretrained network's pick (SSH, SST and MDT, 4 modes)
is the model of the paper. The test sets are never used to choose.

## The model

```python
from swot2argo import data, models, evaluate

tests = data.tests()                                     # {'moorings', 'gliders', 's2'}
model = models.load("results/models/headline_swot.pt")   # SSHA, SST, MDT -> 4 PCA modes -> density profile
dens = model.predict(tests["moorings"])                  # (n, 40) potential density, kg/m3
evaluate.rmse(dens, tests["moorings"].dens)              # 0.130

base = models.Network().pretrain(data.history())         # stage 1: DUACS SSHA, SST, MDT, 14,448 pre-2023 profiles
model = base.finetune(data.pairs(), ssh="swot")          # stage 2: every layer on the 424 SWOT-Argo pairs
```

An ensemble of 15 networks (two hidden layers of 256 units) predicts the coefficients of the four leading PCA modes
of the density anomaly from the ECCO climatology; the anomaly is added to the climatology and the profile projected
onto the statically stable ones. The same recipe fine-tuned and run with DUACS SSHA (`headline_duacs.pt`) is the
comparison of the paper. Any `data.Samples` with `ssh_swot` (or `ssh_duacs`), `sst`, `mdt`, `clim` and `time`
can be predicted; `data.swath_pixels(data.swath(...))` turns a whole SWOT overpass into one (Figure 4).
`models.PairsNetwork` and `models.Ridge` are the other families of the selection.

## Layout

| path | content |
|---|---|
| `swot2argo/config.py` | paths and the fixed choices (inputs, 4 modes, 15 members, folds, bootstrap) |
| `swot2argo/data.py` | loading the data folder and the results |
| `swot2argo/models.py` | the inputs, the PCA target, the ensemble, the two-stage `Network`, `PairsNetwork`, `Ridge`, `save` and `load` |
| `swot2argo/physics.py` | static stability, mixed-layer depth |
| `swot2argo/evaluate.py` | cross-validation and the selection rule; RMSE with block-bootstrap intervals, SWOT against DUACS, persistence, mixed-layer depth, per-mode skill |
| `swot2argo/gradients.py` | horizontal-gradient skill between profile pairs |
| `swot2argo/plots.py` | the figures |
| `results/` | `models/` (the trained network), `predictions.npz`, `configurations.npz` (Table S8), `cv.csv` and `cv_predictions.npz` (the selection), `test_scores.csv`, `gradients.csv`, `tables/`, `summary.txt` |

The data folder was prepared from the public products listed in [../data/README.md](../data/README.md); that
preparation (downloads, colocation of the profiles with the SWOT pixels, calibration of the swath against DUACS, the
crossover statistics of Figure 3a) is not part of this repository.

## Citing

Champenois, B., Ali, A., and Bodner, A. (2026). Reconstructing 3D upper-ocean density from SWOT's wide-swath sea
surface height using machine learning. *Geophysical Research Letters*. Code and data: this repository and its Zenodo
archive. See [../CITATION.cff](../CITATION.cff). The code is under the MIT license ([../LICENSE](../LICENSE)); the
data terms are in [../data/README.md](../data/README.md).
