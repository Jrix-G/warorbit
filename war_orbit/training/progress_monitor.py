"""Bayesian progress monitor: Beta posterior + SPRT promo + GP slope.

Module standalone, zero dependencies hors numpy.

Three responsibilities:
  1. ProgressMonitor.record(gen, mode, opp, wins, n) -> updates Beta posteriors.
  2. ProgressMonitor.sprt_promote(challenger_stats, champion_stats, ...) -> Wald SPRT.
  3. ProgressMonitor.skill_trend(window) -> GP-fit slope + plateau detection.

Mathematical contracts:
  - Each (gen, mode, opp) tuple: Beta(alpha, beta) posterior; alpha0=beta0=0.5 (Jeffreys).
  - Skill posterior per gen = weighted combination over (mode, opp) pairs (importance weights).
  - SPRT: H0: p_chal <= p_champ; H1: p_chal >= p_champ + delta.
    Stops when log-likelihood ratio crosses A=log((1-beta)/alpha) or B=log(beta/(1-alpha)).
  - GP: zero-mean GP with Matern-3/2 kernel on generation index.
    Posterior mean derivative + 1-sigma band -> plateau if |slope| < eps_slope w.p. > 0.9.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class BetaPosterior:
    alpha: float = 0.5
    beta: float = 0.5

    def update(self, wins: int, n: int) -> None:
        wins = int(max(0, wins))
        n = int(max(0, n))
        losses = max(0, n - wins)
        self.alpha += wins
        self.beta += losses

    @property
    def mean(self) -> float:
        return self.alpha / max(1e-12, self.alpha + self.beta)

    @property
    def var(self) -> float:
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / max(1e-12, s * s * (s + 1.0))

    def quantile(self, q: float) -> float:
        # Beta inverse CDF via numerical bracketing (no scipy).
        from math import lgamma, exp
        a, b = self.alpha, self.beta

        def cdf(x: float) -> float:
            # Regularized incomplete beta via continued fraction (Lentz).
            if x <= 0.0:
                return 0.0
            if x >= 1.0:
                return 1.0
            ln_norm = lgamma(a + b) - lgamma(a) - lgamma(b)
            front = exp(ln_norm + a * math.log(x) + b * math.log(1.0 - x)) / a
            # Continued fraction
            fpmin = 1e-300
            qab = a + b
            qap = a + 1.0
            qam = a - 1.0
            c = 1.0
            d = 1.0 - qab * x / qap
            if abs(d) < fpmin:
                d = fpmin
            d = 1.0 / d
            h = d
            for m in range(1, 200):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d
                if abs(d) < fpmin:
                    d = fpmin
                c = 1.0 + aa / c
                if abs(c) < fpmin:
                    c = fpmin
                d = 1.0 / d
                h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d
                if abs(d) < fpmin:
                    d = fpmin
                c = 1.0 + aa / c
                if abs(c) < fpmin:
                    c = fpmin
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < 1e-10:
                    break
            return front * h

        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if cdf(mid) < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def lcb(self, alpha: float = 0.05) -> float:
        return self.quantile(alpha)

    def ucb(self, alpha: float = 0.05) -> float:
        return self.quantile(1.0 - alpha)


def sprt_decision(
    wins_chal: int, n_chal: int,
    p_champ: float,
    delta: float = 0.05,
    alpha: float = 0.05,
    beta: float = 0.10,
) -> Tuple[str, float]:
    """Wald SPRT.

    H0: p = p_champ; H1: p = p_champ + delta.
    Returns ("accept" | "reject" | "continue", log_lr).
    """
    p0 = max(1e-6, min(1 - 1e-6, p_champ))
    p1 = max(1e-6, min(1 - 1e-6, p_champ + delta))
    losses = max(0, n_chal - wins_chal)
    log_lr = (
        wins_chal * math.log(p1 / p0) + losses * math.log((1 - p1) / (1 - p0))
    )
    A = math.log((1.0 - beta) / alpha)
    B = math.log(beta / (1.0 - alpha))
    if log_lr >= A:
        return "accept", log_lr
    if log_lr <= B:
        return "reject", log_lr
    return "continue", log_lr


def gp_slope(
    gens: np.ndarray, skills: np.ndarray, sigmas: np.ndarray,
    length_scale: float = 4.0,
    signal_var: float = 0.04,
) -> Tuple[float, float]:
    """Fit Matern-3/2 GP, return (slope_mean, slope_std) at the latest generation.

    skills: posterior mean skill per generation.
    sigmas: posterior std per generation (heteroscedastic noise).
    """
    n = len(gens)
    if n < 3:
        return 0.0, 1.0
    x = np.asarray(gens, dtype=np.float64)
    y = np.asarray(skills, dtype=np.float64)
    noise = np.asarray(sigmas, dtype=np.float64) ** 2 + 1e-6

    def matern32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.abs(a[:, None] - b[None, :]) / max(1e-6, length_scale)
        sq3 = math.sqrt(3.0)
        return signal_var * (1.0 + sq3 * d) * np.exp(-sq3 * d)

    K = matern32(x, x) + np.diag(noise)
    try:
        L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y))

    # Derivative of Matern-3/2: dK/dx*  = signal_var * exp(-sq3 d) * (-3 (x*-x_i)/l^2)
    x_star = x[-1]
    diff = x_star - x  # shape (n,)
    sq3 = math.sqrt(3.0)
    d = np.abs(diff) / max(1e-6, length_scale)
    dk_dxstar = signal_var * np.exp(-sq3 * d) * (-3.0 * diff / (length_scale ** 2))
    slope_mean = float(dk_dxstar @ alpha_vec)
    v = np.linalg.solve(L, dk_dxstar)
    # k**(x*,x*) for derivative = signal_var * 3 / length_scale^2
    var_prior = signal_var * 3.0 / (length_scale ** 2)
    slope_var = max(1e-12, var_prior - float(v @ v))
    return slope_mean, math.sqrt(slope_var)


@dataclass
class GenStats:
    generation: int
    posteriors: Dict[str, BetaPosterior] = field(default_factory=dict)  # key = f"{mode}|{opp}"
    weights: Dict[str, float] = field(default_factory=dict)

    def record(self, mode: str, opp: str, wins: int, n: int, weight: float = 1.0) -> None:
        key = f"{mode}|{opp}"
        if key not in self.posteriors:
            self.posteriors[key] = BetaPosterior()
            self.weights[key] = float(weight)
        self.posteriors[key].update(wins, n)

    def aggregated_skill(self) -> Tuple[float, float]:
        if not self.posteriors:
            return 0.0, 1.0
        means, vars_, ws = [], [], []
        for key, post in self.posteriors.items():
            w = float(self.weights.get(key, 1.0))
            means.append(post.mean)
            vars_.append(post.var)
            ws.append(w)
        ws = np.asarray(ws, dtype=np.float64)
        ws = ws / max(1e-12, ws.sum())
        m = float(np.sum(ws * np.asarray(means)))
        v = float(np.sum((ws ** 2) * np.asarray(vars_)))
        return m, math.sqrt(max(1e-12, v))


@dataclass
class ProgressMonitor:
    history: List[GenStats] = field(default_factory=list)
    log_path: Optional[str] = None

    def record(self, generation: int, mode: str, opp: str, wins: int, n: int, weight: float = 1.0) -> None:
        if self.history and self.history[-1].generation == generation:
            stats = self.history[-1]
        else:
            stats = GenStats(generation=generation)
            self.history.append(stats)
        stats.record(mode, opp, wins, n, weight)

    def latest_skill(self) -> Tuple[float, float]:
        if not self.history:
            return 0.0, 1.0
        return self.history[-1].aggregated_skill()

    def skill_series(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gens = np.array([h.generation for h in self.history], dtype=np.float64)
        means_sigmas = [h.aggregated_skill() for h in self.history]
        means = np.array([ms[0] for ms in means_sigmas], dtype=np.float64)
        sigmas = np.array([ms[1] for ms in means_sigmas], dtype=np.float64)
        return gens, means, sigmas

    def slope(self, length_scale: float = 4.0) -> Tuple[float, float]:
        gens, means, sigmas = self.skill_series()
        return gp_slope(gens, means, sigmas, length_scale=length_scale)

    def is_plateau(self, eps_slope: float = 0.005, confidence: float = 0.9) -> bool:
        gens, means, sigmas = self.skill_series()
        if len(gens) < 5:
            return False
        slope_mu, slope_sd = gp_slope(gens, means, sigmas)
        # P(|slope| < eps) under N(slope_mu, slope_sd^2)
        from math import erf, sqrt
        def phi(z): return 0.5 * (1.0 + erf(z / sqrt(2.0)))
        p_below = phi((eps_slope - slope_mu) / max(1e-9, slope_sd)) - phi((-eps_slope - slope_mu) / max(1e-9, slope_sd))
        return p_below >= confidence

    def promote_decision(
        self,
        challenger_wins: int, challenger_n: int,
        delta: float = 0.05, alpha: float = 0.05, beta: float = 0.10,
        champion_skill: Optional[float] = None,
    ) -> Tuple[str, float]:
        if champion_skill is None:
            # Use posterior mean of previous best generation as champion if available.
            if len(self.history) >= 2:
                champion_skill = self.history[-2].aggregated_skill()[0]
            else:
                champion_skill = 0.5
        return sprt_decision(challenger_wins, challenger_n, champion_skill, delta=delta, alpha=alpha, beta=beta)

    def to_jsonl(self, path: Optional[str] = None) -> None:
        target = Path(path or self.log_path or "evaluations/progress_monitor.jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            if not self.history:
                return
            h = self.history[-1]
            mean, sigma = h.aggregated_skill()
            slope_mu, slope_sd = self.slope() if len(self.history) >= 3 else (0.0, 1.0)
            f.write(json.dumps({
                "generation": h.generation,
                "skill_mean": mean,
                "skill_sigma": sigma,
                "slope_mean": slope_mu,
                "slope_sigma": slope_sd,
                "plateau": self.is_plateau(),
                "posteriors": {k: {"alpha": p.alpha, "beta": p.beta} for k, p in h.posteriors.items()},
            }, sort_keys=True) + "\n")
