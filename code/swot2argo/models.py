"""The models of the paper.

The reported model is `Network`: an ensemble of small fully connected networks (Pauthenet et al. 2022) that maps the
surface inputs to the coefficients of the leading PCA modes of the density anomaly, trained in two stages.

    from swot2argo import data, models
    base = models.Network().pretrain(data.history())          # stage 1: DUACS SSHA, SST, MDT on the pre-2023 record
    model = base.finetune(data.pairs(), ssh="swot")           # stage 2: all layers on the SWOT-Argo pairs
    dens = model.predict(data.tests()["moorings"])            # potential density (n, 40), kg/m3, stabilized
    models.save(model, "headline_swot.pt"); model = models.load("headline_swot.pt")

The selection (SI Text S3, Table S8) also compares the same network trained on the pairs alone (`PairsNetwork`), a
ridge regression (`Ridge`) and other inputs. Every model predicts the density ANOMALY from the climatology, on the
first `n_modes` PCA modes or on all 40 levels (`n_modes=None`), and returns the climatology plus that anomaly,
projected onto the statically stable profiles.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FEATURES, N_MEMBERS, N_MODES
from .physics import stabilize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- the inputs
INPUT_NAMES = {"ssh": "SSH", "sst": "SST", "mdt": "MDT", "sss": "SSS", "bathy": "bathymetry",
               "latlon": "latitude-longitude", "era5": "ERA5"}
ERA5_COLUMNS = [f"era5_{v}_{w}" for v in ("b0", "ustar3") for w in (7, 30, 60)]   # buoyancy flux, u*^3; 7-60 d means


def input_matrix(S, features, ssh):
    """The inputs at the profile's pixel, one or more columns per feature: 'ssh' the SSHA of the product `ssh`
    ('swot' or 'duacs'), 'sst' OSTIA SST, 'mdt' the mean dynamic topography; the candidates that were not kept:
    'sss' satellite SSS, 'bathy' the bottom depth, 'latlon' latitude and longitude, 'era5' the ERA5 surface buoyancy
    flux and u*^3 averaged over 7, 30 and 60 days. NaNs stay (see Filler)."""
    if ssh not in ("swot", "duacs"):
        raise ValueError("ssh must be 'swot' or 'duacs'")
    cols = {"ssh": [f"ssh_{ssh}"], "latlon": ["lat", "lon"], "era5": ERA5_COLUMNS}
    return np.column_stack([np.asarray(S[c], float) for f in features for c in cols.get(f, [f])])


def n_columns(features):
    return sum({"latlon": 2, "era5": len(ERA5_COLUMNS)}.get(f, 1) for f in features)


def label(features):
    return " + ".join(INPUT_NAMES[f] for f in features)


class Filler:
    """Fills missing inputs with the medians of the training matrix (SSS has none before June 2010)."""

    def fit(self, X):
        med = np.nanmedian(X, 0)
        self.med = np.where(np.isfinite(med), med, 0.0)
        return self

    def __call__(self, X):
        X = np.array(X, float, copy=True)
        bad = ~np.isfinite(X)
        X[bad] = np.take(self.med, np.where(bad)[1])
        return X


# ----------------------------------------------------------------------------- the target
class Target:
    """Anomaly <-> model output: the coefficients of the first n_modes PCA modes, or with n_modes=None the 40
    levels minus their mean. `all_modes` and `explained` describe the whole basis either way."""

    def __init__(self, n_modes=N_MODES):
        self.n_modes = n_modes

    def fit(self, anom):
        anom = np.asarray(anom, float)
        self.mean = anom.mean(0)
        _, s, vt = np.linalg.svd(anom - self.mean, full_matrices=False)
        sign = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(1)])          # largest value of each mode positive
        self.all_modes = vt * sign[:, None]
        self.explained = s ** 2 / (s ** 2).sum()
        self.modes = self.all_modes[:self.n_modes] if self.n_modes else None
        return self

    def encode(self, anom):
        a = np.asarray(anom, float) - self.mean
        return a if self.modes is None else a @ self.modes.T

    def decode(self, Y):
        return (Y if self.modes is None else Y @ self.modes) + self.mean


# ----------------------------------------------------------------------------- the ensemble
class _Net(nn.Module):
    """Two hidden layers of 256 ReLU units, dropout 0.1 after the first."""

    def __init__(self, n_in, n_out, width=256, dropout=0.1):
        super().__init__()
        self.l1, self.l2, self.head = nn.Linear(n_in, width), nn.Linear(width, width), nn.Linear(width, n_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.head(F.relu(self.l2(F.relu(self.drop(self.l1(x))))))


class Ensemble:
    """Identical networks with different seeds and validation splits, averaged. Max-norm 2 on every weight row,
    Adam, early stopping on the validation loss. Inputs are standardized by column, outputs by one scalar."""

    def __init__(self, n_members=N_MEMBERS, seed=0, lr=1e-4, batch=32, epochs=1000, patience=20, max_norm=2.0,
                 width=256, dropout=0.1, val_frac=0.2):
        self.n_members, self.seed, self.lr, self.batch, self.epochs = n_members, seed, lr, batch, epochs
        self.patience, self.max_norm, self.width, self.dropout, self.val_frac = (patience, max_norm, width,
                                                                                  dropout, val_frac)

    def _renorm(self, net):
        with torch.no_grad():
            for lay in (net.l1, net.l2, net.head):
                n = lay.weight.norm(dim=1, keepdim=True).clamp(min=1e-12)
                lay.weight.mul_(n.clamp(max=self.max_norm) / n)

    def _split(self, groups, k):
        """Validation mask: a fifth of the floats (whole floats, so no float is on both sides)."""
        rng = np.random.default_rng(self.seed + k)
        g = np.unique(groups)
        va = np.isin(groups, rng.choice(g, max(1, int(round(self.val_frac * len(g)))), replace=False))
        if va.all() or not va.any():
            va = np.zeros(len(groups), bool)
            va[rng.choice(len(groups), max(1, len(groups) // 5), replace=False)] = True
        return va

    def _train(self, net, Xs, Ys, va, params, lr, epochs, patience):
        opt = torch.optim.Adam(params, lr=lr)
        itr = torch.where(torch.tensor(~va, device=DEVICE))[0]
        iva = torch.where(torch.tensor(va, device=DEVICE))[0]
        best, bad = np.inf, 0
        state = {q: v.clone() for q, v in net.state_dict().items()}
        for _ in range(epochs):
            net.train()
            idx = itr[torch.randperm(len(itr), device=DEVICE)]
            for b in range(0, len(idx), self.batch):
                s = idx[b:b + self.batch]
                opt.zero_grad()
                F.mse_loss(net(Xs[s]), Ys[s]).backward()
                opt.step()
                self._renorm(net)
            net.eval()
            with torch.no_grad():
                vl = F.mse_loss(net(Xs[iva]), Ys[iva]).item()
            if vl < best - 1e-7:
                best, bad, state = vl, 0, {q: v.clone() for q, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(state)
        return net

    def _tensors(self, X, Y=None):
        Xs = torch.tensor((np.asarray(X, float) - self.xmu) / self.xsd, dtype=torch.float32, device=DEVICE)
        if Y is None:
            return Xs
        return Xs, torch.tensor((np.asarray(Y, float) - self.ymu) / self.ysd, dtype=torch.float32, device=DEVICE)

    def fit(self, X, Y, groups):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-9
        self.ymu, self.ysd = Y.mean(), Y.std() + 1e-9
        Xs, Ys = self._tensors(X, Y)
        self.nets = []
        for k in range(self.n_members):
            torch.manual_seed(self.seed + k)
            net = _Net(X.shape[1], Y.shape[1], self.width, self.dropout).to(DEVICE)
            self.nets.append(self._train(net, Xs, Ys, self._split(groups, k), net.parameters(), self.lr,
                                         self.epochs, self.patience))
        return self

    def finetune(self, X, Y, groups, lr=3e-5, epochs=300, patience=25, seed=100):
        """A copy whose layers are all retrained on (X, Y); the input and output scalings are kept."""
        out = copy.deepcopy(self)
        Xs, Ys = out._tensors(X, Y)
        for k, net in enumerate(out.nets):
            tmp = copy.copy(out)
            tmp.seed = seed + k
            out._train(net, Xs, Ys, tmp._split(groups, 0), net.parameters(), lr, epochs, patience)
        return out

    def predict(self, X):
        Xs = self._tensors(X)
        with torch.no_grad():
            ys = np.stack([net.eval()(Xs).cpu().numpy() for net in self.nets]) * self.ysd + self.ymu
        return ys.mean(0)


# ----------------------------------------------------------------------------- the models
class _Model:
    """What every model shares: the inputs it reads and the step from predicted anomaly to density."""
    ssh = None

    def _inputs(self, S, ssh=None):
        return self.filler(input_matrix(S, self.features, ssh or self.ssh))

    def predict_anomaly(self, S, ssh=None):
        """The density anomaly from the climatology (n, 40)."""
        return self.target.decode(self._predict(self._inputs(S, ssh)))

    def predict(self, S, ssh=None):
        """Potential density (n, 40), kg/m3: climatology + anomaly, projected onto the stable profiles."""
        return stabilize(S.clim + self.predict_anomaly(S, ssh))

    def _output(self):
        return "40 levels" if self.n_modes is None else f"{self.n_modes} mode" + "s" * (self.n_modes > 1)


class Network(_Model):
    """The paper's model. Stage 1 (`pretrain`) on the historical record with DUACS SSHA; stage 2 (`finetune`) on
    the SWOT pairs, fed SWOT or DUACS. The PCA basis, the input scaling and the median fill come from stage 1."""

    def __init__(self, features=FEATURES, n_modes=N_MODES, n_members=N_MEMBERS, seed=0):
        self.features, self.n_modes, self.n_members, self.seed = tuple(features), n_modes, n_members, seed

    def __repr__(self):
        stage = "untrained" if self.ssh is None else ("pretrained on DUACS" if self.ssh == "base" else
                                                      f"pretrained on DUACS, fine-tuned on {self.ssh.upper()}")
        return f"<Network {label(self.features)} -> {self._output()}, {self.n_members} members, {stage}>"

    def pretrain(self, history, lr=3e-4, batch=128, verbose=True):
        X = input_matrix(history, self.features, "duacs")
        self.filler = Filler().fit(X)
        self.target = Target(self.n_modes).fit(history.anom)
        if verbose:
            print(f"stage 1: {self.n_members} networks on {history.n} profiles ({len(np.unique(history.wmo))} floats)",
                  flush=True)
        self.ens = Ensemble(n_members=self.n_members, seed=self.seed, lr=lr, batch=batch).fit(
            self.filler(X), self.target.encode(history.anom), np.asarray(history.wmo))
        self.ssh = "base"
        return self

    def finetune(self, pairs, ssh="swot", lr=3e-5):
        """A fine-tuned copy of the pretrained model, fed `ssh` ('swot' or 'duacs') from now on."""
        out = copy.copy(self)
        X = self.filler(input_matrix(pairs, self.features, ssh))
        out.ens = self.ens.finetune(X, self.target.encode(pairs.anom), np.asarray(pairs.wmo), lr=lr,
                                    seed=self.seed + 100)
        out.ssh = ssh
        return out

    def _inputs(self, S, ssh=None):
        return super()._inputs(S, ssh or ("duacs" if self.ssh == "base" else None))

    def _predict(self, X):
        return self.ens.predict(X)


class PairsNetwork(_Model):
    """The same ensemble trained on the SWOT-Argo pairs alone (learning rate 1e-4, batch 32); its PCA basis, input
    scaling and fill come from those pairs."""

    def __init__(self, features=FEATURES, n_modes=N_MODES, n_members=N_MEMBERS, seed=0):
        self.features, self.n_modes, self.n_members, self.seed = tuple(features), n_modes, n_members, seed

    def __repr__(self):
        return f"<PairsNetwork {label(self.features)} -> {self._output()}, {self.n_members} members>"

    def fit(self, pairs, ssh="swot"):
        X = input_matrix(pairs, self.features, ssh)
        self.filler, self.target, self.ssh = Filler().fit(X), Target(self.n_modes).fit(pairs.anom), ssh
        self.ens = Ensemble(n_members=self.n_members, seed=self.seed).fit(
            self.filler(X), self.target.encode(pairs.anom), np.asarray(pairs.wmo))
        return self

    def _predict(self, X):
        return self.ens.predict(X)


class Ridge(_Model):
    """Ridge regression from the standardized inputs to the mode coefficients of the pairs' PCA basis."""

    def __init__(self, features=FEATURES, n_modes=N_MODES, alpha=10.0):
        self.features, self.n_modes, self.alpha = tuple(features), n_modes, alpha

    def __repr__(self):
        return f"<Ridge {label(self.features)} -> {self._output()}, alpha {self.alpha:g}>"

    def fit(self, pairs, ssh="swot"):
        X = input_matrix(pairs, self.features, ssh)
        self.filler, self.target, self.ssh = Filler().fit(X), Target(self.n_modes).fit(pairs.anom), ssh
        X, Y = self.filler(X), self.target.encode(pairs.anom)
        self.xmu, self.xsd = X.mean(0), X.std(0)
        self.xsd[self.xsd == 0] = 1.0
        Z = (X - self.xmu) / self.xsd
        self.ymu = Y.mean(0)
        self.coef = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (Y - self.ymu))
        return self

    def _predict(self, X):
        return ((X - self.xmu) / self.xsd) @ self.coef + self.ymu


# ----------------------------------------------------------------------------- saving and loading
def save(model, path):
    """A Network as plain tensors and numbers, so the file loads with torch's weights_only=True."""
    e = model.ens
    t = lambda a: torch.tensor(np.asarray(a, float))         # noqa: E731
    torch.save(dict(features=list(model.features), n_modes=model.n_modes, n_members=model.n_members, seed=model.seed,
                    ssh=model.ssh, fill=t(model.filler.med), target_mean=t(model.target.mean),
                    target_modes=t(model.target.all_modes[:model.n_modes]), target_explained=t(model.target.explained),
                    xmu=t(e.xmu), xsd=t(e.xsd), ymu=float(e.ymu), ysd=float(e.ysd), width=e.width,
                    dropout=e.dropout, nets=[{k: v.cpu() for k, v in n.state_dict().items()} for n in e.nets]),
               path)


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    m = Network(d["features"], d["n_modes"], d["n_members"], d["seed"])
    m.ssh = d["ssh"]
    m.filler = Filler()
    m.filler.med = d["fill"].numpy()
    m.target = Target(d["n_modes"])
    m.target.mean, m.target.explained = d["target_mean"].numpy(), d["target_explained"].numpy()
    m.target.all_modes = d["target_modes"].numpy()
    m.target.modes = m.target.all_modes if m.n_modes else None
    e = Ensemble(n_members=d["n_members"], seed=d["seed"], width=d["width"], dropout=d["dropout"])
    e.xmu, e.xsd, e.ymu, e.ysd = d["xmu"].numpy(), d["xsd"].numpy(), d["ymu"], d["ysd"]
    e.nets = []
    for sd in d["nets"]:
        net = _Net(sd["l1.weight"].shape[1], sd["head.weight"].shape[0], d["width"], d["dropout"])
        net.load_state_dict(sd)
        e.nets.append(net.to(DEVICE).eval())
    m.ens = e
    return m
