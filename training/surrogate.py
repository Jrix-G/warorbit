"""Surrogate model pour accélérer CMA-ES.

Idée : entraîner un Gaussian Process sur les paires (weights, score) déjà
évaluées. À chaque génération CMA-ES, sampler 5-10x plus de candidats que
la popsize, prédire leur score via le GP, et n'évaluer en réel que le top-k
+ quelques candidats d'exploration (haute variance prédite).

Implémentation : si scikit-learn dispo, on l'utilise (RBF kernel). Sinon
fallback numpy minimal (interpolation par k-NN pondéré). L'API reste la
même.
"""

import math
import numpy as np


class _NumpyKNN:
    """Fallback ultra-simple si sklearn absent : k-NN gaussien pondéré."""

    def __init__(self, k=5, length_scale=0.5):
        self.k = k
        self.length_scale = length_scale
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        if self.X is None or len(self.X) == 0:
            mu = np.zeros(len(X))
            sd = np.ones(len(X))
            return (mu, sd) if return_std else mu

        d = np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2)
        w = np.exp(-(d / self.length_scale) ** 2)
        w_sum = w.sum(axis=1, keepdims=True) + 1e-9
        w_norm = w / w_sum

        mu = (w_norm * self.y[None, :]).sum(axis=1)
        if not return_std:
            return mu
        # std = écart-type pondéré local
        var = (w_norm * (self.y[None, :] - mu[:, None]) ** 2).sum(axis=1)
        # Plus on est loin des points connus, plus l'incertitude monte
        density = w_sum.squeeze(axis=1) / max(len(self.X), 1)
        unc = 1.0 / (1.0 + density)
        sd = np.sqrt(var + 1e-6) + 0.3 * unc
        return mu, sd


class Surrogate:
    """GP wrapper avec fallback numpy."""

    def __init__(self):
        self.X = []
        self.y = []
        self._model = None
        self._sklearn = None
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel
            self._sklearn = (GaussianProcessRegressor, Matern, ConstantKernel)
        except ImportError:
            self._sklearn = None

    def _build(self):
        if self._sklearn is None:
            self._model = _NumpyKNN(k=8, length_scale=0.5)
        else:
            GPR, Matern, Const = self._sklearn
            kernel = Const(1.0, (1e-3, 1e3)) * Matern(length_scale=0.5, nu=2.5)
            self._model = GPR(
                kernel=kernel,
                normalize_y=True,
                alpha=1e-3,
                n_restarts_optimizer=2,
            )

    def add(self, weights, score):
        self.X.append(list(weights))
        self.y.append(float(score))

    def fit(self):
        if len(self.X) < 3:
            return
        self._build()
        X = np.asarray(self.X, dtype=float)
        y = np.asarray(self.y, dtype=float)
        try:
            self._model.fit(X, y)
        except Exception as e:  # GP peut planter si data mal conditionnée
            print(f"  surrogate fit failed ({e}); falling back to KNN")
            self._model = _NumpyKNN()
            self._model.fit(X, y)

    def predict(self, candidates, return_std=True):
        if self._model is None or len(self.X) < 3:
            n = len(candidates)
            return np.zeros(n), np.ones(n)
        X = np.asarray(candidates, dtype=float)
        try:
            return self._model.predict(X, return_std=True)
        except TypeError:
            # GP sans return_std: refaire avec std=False
            return self._model.predict(X, return_std=False), np.zeros(len(X))

    def select_promising(self, candidates, k_keep, k_explore=2):
        """Renvoie indices des candidats à évaluer en réel.

        k_keep = top-k par moyenne prédite (exploitation)
        k_explore = candidats supplémentaires à plus haute std (exploration)
        """
        mu, sd = self.predict(candidates)
        n = len(candidates)
        if n <= k_keep + k_explore:
            return list(range(n))
        order_mu = np.argsort(-mu)  # tri décroissant
        keep = list(order_mu[:k_keep])
        remaining = [i for i in range(n) if i not in set(keep)]
        order_sd = sorted(remaining, key=lambda i: -sd[i])
        keep += order_sd[:k_explore]
        return keep

    def __len__(self):
        return len(self.X)
