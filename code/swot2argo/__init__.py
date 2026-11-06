"""swot2argo: upper-ocean density profiles (0-200 m) from SWOT sea surface height, trained on Argo.

    from swot2argo import data, models, evaluate
    tests = data.tests()                                   # {'moorings', 'gliders', 's2'}
    model = models.load("results/models/headline_swot.pt") # the paper's network, fine-tuned on SWOT
    dens = model.predict(tests["moorings"])                # (n, 40) potential density, kg/m3
    evaluate.rmse(dens, tests["moorings"].dens)

Modules: config (paths, fixed choices), data (the data folder and the results), models (inputs, the network),
physics (stability, mixed-layer depth), evaluate (cross-validation, selection, scores), gradients (horizontal-gradient
skill), plots (the figures).
"""
from . import config, data, evaluate, gradients, models, physics  # noqa: F401
from .config import DEPTH  # noqa: F401
