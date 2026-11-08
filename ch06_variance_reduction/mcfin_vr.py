"""mcfin_vr.py -- tools for variance reduction (Chapter 6)."""
import time
import numpy as np
from scipy.stats import norm


def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))


def summary(Y, n_payoffs=None):
    """Estimate, standard error and variance per payoff evaluation of i.i.d. Y.

    n_payoffs is the number of payoff evaluations used to produce Y
    (2 * len(Y) for antithetic pairs); by default len(Y)."""
    n = Y.size
    n_payoffs = n if n_payoffs is None else n_payoffs
    se = Y.std(ddof=1) / np.sqrt(n)
    return Y.mean(), se, se**2 * n_payoffs        # last: variance per evaluation


class Timer:
    """Context manager measuring wall-clock time: with Timer() as t: ...; t.s"""
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.s = time.perf_counter() - self.t0


def control_variate(Y, X, mean_X):
    """Control-variate estimate with the optimal (estimated) coefficient b.

    X may be one control (vector) or several (matrix with one column each)."""
    X = np.asarray(X, dtype=float).reshape(Y.size, -1)
    Xc = X - X.mean(axis=0)
    b = np.linalg.lstsq(Xc, Y - Y.mean(), rcond=None)[0]   # regression slope(s)
    Y_cv = Y - (X - np.asarray(mean_X)) @ b
    return Y_cv, b
