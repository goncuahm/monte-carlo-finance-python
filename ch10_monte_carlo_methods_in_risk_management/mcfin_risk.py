"""mcfin_risk.py -- risk measures, option books and backtests (Chapter 10)."""
import numpy as np
from scipy.stats import norm, chi2


def var_es(losses, alpha):
    """Value-at-Risk and Expected Shortfall of simulated losses (positive = loss).

    ES is the average of the worst ceil((1 - alpha) N) losses, which is correct
    also for discrete loss distributions."""
    n = losses.size
    k = int(np.ceil((1 - alpha) * n))
    worst = np.partition(losses, n - k)[n - k:]
    return np.quantile(losses, alpha), worst.mean()


def bootstrap_ci(losses, alpha, rng, n_boot=500, level=0.95):
    """Bootstrap confidence intervals for VaR and ES."""
    n = losses.size
    stats = np.array([var_es(losses[rng.integers(0, n, n)], alpha) for _ in range(n_boot)])
    q = [(1 - level) / 2, (1 + level) / 2]
    return np.quantile(stats[:, 0], q), np.quantile(stats[:, 1], q)


def bs_value(S, K, r, sigma, tau, kind):
    """Black-Scholes call/put value for array S and maturity tau > 0."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    call = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return call if kind == "call" else call - S + K * np.exp(-r * tau)


def bs_delta_gamma(S, K, r, sigma, tau, kind):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    delta = norm.cdf(d1) - (0 if kind == "call" else 1)
    return delta, norm.pdf(d1) / (S * sigma * np.sqrt(tau))


def kupiec(exceptions, n, p):
    """Kupiec proportion-of-failures test: LR statistic and p-value."""
    x = exceptions
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + (x * np.log(phat) if x > 0 else 0.0)
    lr = -2 * (ll0 - ll1)
    return lr, chi2.sf(lr, df=1)
