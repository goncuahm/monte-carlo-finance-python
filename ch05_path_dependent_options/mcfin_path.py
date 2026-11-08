"""mcfin_path.py -- path-dependent options (Chapter 5)."""
import numpy as np
from scipy.stats import norm


def mc_price(disc_payoff):
    """Monte Carlo estimate and standard error from discounted payoffs."""
    n = disc_payoff.size
    return disc_payoff.mean(), disc_payoff.std(ddof=1) / np.sqrt(n)


def gbm_steps(rng, S0, r, sigma, T, n_steps, n_paths):
    """Generator yielding S(t_1), ..., S(t_n) one step at a time (saves memory)."""
    dt = T / n_steps
    drift, vol = (r - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt)
    logS = np.full(n_paths, np.log(S0))
    for _ in range(n_steps):
        logS = logS + drift + vol * rng.standard_normal(n_paths)
        yield np.exp(logS)


def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))


def geometric_asian_call(S0, K, r, sigma, T, n):
    """Exact price of a call on the geometric average of S(T/n), ..., S(T)."""
    m = np.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d1 = (m - np.log(K) + v) / np.sqrt(v)
    return np.exp(-r * T) * (np.exp(m + v / 2) * norm.cdf(d1)
                             - K * norm.cdf(d1 - np.sqrt(v)))


def down_out_call(S0, K, B, r, sigma, T):
    """Continuously monitored down-and-out call, barrier B <= K, no rebate."""
    lam = (r + 0.5 * sigma**2) / sigma**2
    y = np.log(B**2 / (S0 * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
    c_di = (S0 * (B / S0)**(2 * lam) * norm.cdf(y)
            - K * np.exp(-r * T) * (B / S0)**(2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T)))
    return bs_call(S0, K, r, sigma, T) - c_di


def lookback_call(S0, r, sigma, T):
    """Floating-strike lookback call paying S(T) - min S, continuous monitoring."""
    a1 = (r + 0.5 * sigma**2) * np.sqrt(T) / sigma
    a2, a3 = a1 - sigma * np.sqrt(T), (-r + 0.5 * sigma**2) * np.sqrt(T) / sigma
    k = sigma**2 / (2 * r)
    return (S0 * norm.cdf(a1) - S0 * k * norm.cdf(-a1)
            - S0 * np.exp(-r * T) * (norm.cdf(a2) - k * norm.cdf(-a3)))
