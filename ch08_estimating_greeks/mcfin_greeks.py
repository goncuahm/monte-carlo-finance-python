"""mcfin_greeks.py -- Black-Scholes Greeks and helpers (Chapter 8)."""
import numpy as np
from scipy.stats import norm


def bs_greeks(S0, K, r, sigma, T):
    """Price, delta, gamma and vega of a European call (Black-Scholes)."""
    sT = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sT
    d2 = d1 - sT
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return {"price": price, "delta": norm.cdf(d1),
            "gamma": norm.pdf(d1) / (S0 * sT), "vega": S0 * norm.pdf(d1) * np.sqrt(T)}


def digital_greeks(S0, K, r, sigma, T):
    """Price and delta of a digital call paying 1 if S(T) > K."""
    sT = sigma * np.sqrt(T)
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / sT
    return {"price": np.exp(-r * T) * norm.cdf(d2),
            "delta": np.exp(-r * T) * norm.pdf(d2) / (S0 * sT)}


def terminal(S0, r, sigma, T, Z):
    """S(T) under Black-Scholes from standard normals Z."""
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def mean_se(Y):
    return Y.mean(), Y.std(ddof=1) / np.sqrt(Y.size)
