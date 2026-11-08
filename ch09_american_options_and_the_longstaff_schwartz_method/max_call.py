import numpy as np
from mcfin_amer import lsm, apply_policy

# Bermudan max-call on two assets (Broadie-Glasserman / Andersen-Broadie benchmark):
# S0 = 100 each, K = 100, r = 5%, dividend yield 10%, sigma = 20%, rho = 0, T = 3,
# 9 exercise dates. Published value about 13.90.
S0, K, r, q, sigma, T, M = 100.0, 100.0, 0.05, 0.10, 0.20, 3.0, 9
dt = T / M
def paths(rng, N):
    Z = rng.standard_normal((N, M, 2))
    logS = np.log(S0) + np.cumsum((r - q - 0.5 * sigma**2) * dt
                                  + sigma * np.sqrt(dt) * Z, axis=1)
    S = np.concatenate([np.full((N, 1, 2), S0), np.exp(logS)], axis=1)
    return S                                              # shape N x (M+1) x 2

def features(S):                    # basis in the two prices, scaled by K
    x1, x2 = np.sort(S, axis=-1).T / K                    # x2 = larger price
    return np.column_stack([np.ones_like(x1), x2, x2**2, x2**3, x1, x1**2,
                            x1 * x2, np.maximum(x2 - 1, 0), np.maximum(x2 - 1, 0) * x1])

payoff = lambda S: np.maximum(S.max(axis=-1) - K, 0)
def lsm2(S):
    N = S.shape[0]
    cash, tau, coefs = payoff(S[:, M]), np.full(N, M), [None] * (M + 1)
    for j in range(M - 1, 0, -1):
        h = payoff(S[:, j]); itm = h > 0
        X = features(S[itm, j]); Y = cash[itm] * np.exp(-r * dt * (tau[itm] - j))
        coefs[j] = np.linalg.lstsq(X, Y, rcond=None)[0]
        ex = np.where(itm)[0][h[itm] > X @ coefs[j]]
        cash[ex], tau[ex] = h[ex], j
    return np.mean(cash * np.exp(-r * dt * tau)), coefs
def policy(S, coefs):
    N = S.shape[0]; value, done = np.zeros(N), np.zeros(N, dtype=bool)
    for j in range(1, M + 1):
        h = payoff(S[:, j])
        ex = ~done if j == M else ~done & (h > 0) & (h > features(S[:, j]) @ coefs[j])
        value[ex] = h[ex] * np.exp(-r * dt * j); done |= ex
    return value
p_in, coefs = lsm2(paths(np.random.default_rng(1), 200_000))
v = policy(paths(np.random.default_rng(2), 200_000), coefs)
v_eu = np.exp(-r * T) * payoff(paths(np.random.default_rng(2), 200_000)[:, M])
print(f"in-sample LSM        {p_in:.4f}")
print(f"out-of-sample LSM    {v.mean():.4f}  s.e. {v.std()/np.sqrt(v.size):.4f}")
print(f"European max-call    {v_eu.mean():.4f}  s.e. {v_eu.std()/np.sqrt(v_eu.size):.4f}")
