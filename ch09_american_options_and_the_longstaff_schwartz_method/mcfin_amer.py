"""mcfin_amer.py -- American and Bermudan options (Chapter 9)."""
import numpy as np
from scipy.stats import norm


def bs_put(S, K, r, sigma, tau):
    """European put; works for arrays and returns the payoff when tau = 0."""
    tau = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)


def binomial_put(S0, K, r, sigma, T, n, american=True, boundary=False):
    """Cox-Ross-Rubinstein tree for a put; optionally the exercise boundary."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    S = S0 * u**np.arange(n, -n - 1, -2)              # terminal prices, high to low
    V = np.maximum(K - S, 0)
    bnd = np.full(n + 1, np.nan); bnd[n] = K
    for k in range(n - 1, -1, -1):
        S = S0 * u**np.arange(k, -k - 1, -2)
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            ex = K - S
            exercise = ex > V
            V = np.where(exercise, ex, V)
            if boundary and exercise.any():
                bnd[k] = S[exercise].max()          # highest price with exercise
    return (V[0], bnd) if boundary else V[0]


def gbm_paths(rng, S0, r, sigma, T, M, N, q=0.0):
    """N paths on M equally spaced dates (column 0 is t = 0)."""
    dt = T / M
    Z = rng.standard_normal((N, M))
    logS = np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    return np.hstack([np.full((N, 1), S0), S0 * np.exp(logS)])


def basis(x, degree):
    """Polynomial basis 1, x, ..., x^degree (x should be scaled, e.g. S/K)."""
    return np.vander(x, degree + 1, increasing=True)


def lsm(S, payoff, r, dt, degree=3, scale=1.0, itm_only=True):
    """Longstaff-Schwartz on simulated paths S (N x (M+1)).

    Returns the estimated price and the regression coefficients per date,
    which define an exercise policy that can be applied to new paths."""
    N, M1 = S.shape
    M = M1 - 1
    cash = payoff(S[:, M])                     # cash flow along each path ...
    tau = np.full(N, M)                        # ... received at date tau
    coefs = [None] * M1
    for j in range(M - 1, 0, -1):
        h = payoff(S[:, j])
        rows = (h > 0) if itm_only else np.ones(N, dtype=bool)
        if rows.sum() <= degree:
            continue
        Y = cash * np.exp(-r * dt * (tau - j))  # discounted future cash flow
        X = basis(S[rows, j] / scale, degree)
        beta = np.linalg.lstsq(X, Y[rows], rcond=None)[0]
        cont = X @ beta                          # estimated continuation value
        ex = h[rows] > cont
        idx = np.where(rows)[0][ex]
        cash[idx], tau[idx] = h[idx], j
        coefs[j] = beta
    price = np.mean(cash * np.exp(-r * dt * tau))
    return max(price, payoff(np.array([S[0, 0]]))[0]), coefs


def apply_policy(S, payoff, r, dt, coefs, degree=3, scale=1.0):
    """Exercise according to given regression coefficients (out-of-sample)."""
    N, M1 = S.shape
    M = M1 - 1
    value, done = np.zeros(N), np.zeros(N, dtype=bool)
    for j in range(1, M + 1):
        h = payoff(S[:, j])
        if j == M:
            ex = ~done
        elif coefs[j] is None:
            continue
        else:
            cont = basis(S[:, j] / scale, degree) @ coefs[j]
            ex = ~done & (h > 0) & (h > cont)
        value[ex] = h[ex] * np.exp(-r * dt * j)
        done |= ex
    return value
