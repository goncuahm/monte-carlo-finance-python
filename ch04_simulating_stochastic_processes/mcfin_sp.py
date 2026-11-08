"""mcfin_sp.py -- simulation of stochastic processes (Chapter 4)."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from math import factorial


def brownian_paths(rng, T, n_steps, n_paths):
    """Time grid t and Brownian paths W (n_paths x (n_steps+1)), W[:, 0] = 0."""
    dt = T / n_steps
    dW = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    return np.linspace(0.0, T, n_steps + 1), W


def gbm_paths(rng, S0, mu, sigma, T, n_steps, n_paths):
    """Exact simulation of geometric Brownian motion on a uniform grid."""
    t, W = brownian_paths(rng, T, n_steps, n_paths)
    return t, S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)


def euler_maruyama(a, b, x0, T, dW):
    """Euler scheme for dX = a(t,X) dt + b(t,X) dW, given increments dW (paths x steps)."""
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    X = np.full(n_paths, float(x0))
    for k in range(n_steps):
        t = k * dt
        X = X + a(t, X) * dt + b(t, X) * dW[:, k]
    return X


def poisson_arrivals(rng, lam, T):
    """Jump times of a Poisson process with intensity lam on [0, T]."""
    times = []
    t = rng.exponential(1 / lam)
    while t <= T:
        times.append(t)
        t += rng.exponential(1 / lam)
    return np.array(times)


def merton_paths(rng, S0, r, sigma, lam, muJ, delJ, T, n_steps, n_paths):
    """Risk-neutral Merton jump-diffusion paths; log jump sizes are N(muJ, delJ^2)."""
    dt = T / n_steps
    kappa = np.exp(muJ + 0.5 * delJ**2) - 1            # mean relative jump size
    drift = (r - lam * kappa - 0.5 * sigma**2) * dt
    Z = rng.standard_normal((n_paths, n_steps))
    N = rng.poisson(lam * dt, size=(n_paths, n_steps))  # number of jumps per step
    J = muJ * N + delJ * np.sqrt(N) * rng.standard_normal((n_paths, n_steps))
    logS = np.log(S0) + np.cumsum(drift + sigma * np.sqrt(dt) * Z + J, axis=1)
    S = np.concatenate([np.full((n_paths, 1), S0), np.exp(logS)], axis=1)
    return np.linspace(0.0, T, n_steps + 1), S


def bs_call(S0, K, r, sigma, T):
    """Black-Scholes price of a European call."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))


def implied_vol(price, S0, K, r, T):
    """Black-Scholes implied volatility of a call price (root finding)."""
    return brentq(lambda s: bs_call(S0, K, r, s, T) - price, 1e-4, 5.0)


def merton_call(S0, K, r, sigma, lam, muJ, delJ, T, n_terms=60):
    """Merton (1976) series formula: a Poisson mixture of Black-Scholes prices."""
    kappa = np.exp(muJ + 0.5 * delJ**2) - 1
    lam2 = lam * (1 + kappa)
    price = 0.0
    for n in range(n_terms):
        sig_n = np.sqrt(sigma**2 + n * delJ**2 / T)
        r_n = r - lam * kappa + n * np.log(1 + kappa) / T
        w = np.exp(-lam2 * T) * (lam2 * T)**n / factorial(n)
        price += w * bs_call(S0, K, r_n, sig_n, T)
    return price
