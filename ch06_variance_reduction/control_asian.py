import numpy as np
from mcfin_vr import summary, control_variate
from scipy.stats import norm

# Arithmetic Asian call, monthly averaging; control = geometric Asian call (Chapter 5)
S0, K, r, sigma, T, n, N = 100.0, 100.0, 0.05, 0.30, 1.0, 12, 100_000
m = np.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
d1 = (m - np.log(K) + v) / np.sqrt(v)
geo_exact = np.exp(-r * T) * (np.exp(m + v / 2) * norm.cdf(d1) - K * norm.cdf(d1 - np.sqrt(v)))

rng = np.random.default_rng(seed=4)
dt = T / n
logS = np.log(S0) + np.cumsum((r - 0.5 * sigma**2) * dt
                              + sigma * np.sqrt(dt) * rng.standard_normal((N, n)), axis=1)
S = np.exp(logS)
disc = np.exp(-r * T)
Y = disc * np.maximum(S.mean(axis=1) - K, 0)                 # arithmetic payoff
G = disc * np.maximum(np.exp(logS.mean(axis=1)) - K, 0)      # geometric payoff
X2 = disc * S[:, -1]                                          # discounted S(T), mean S0

for label, X, mX in [("none", None, None),
                     ("S(T)", X2, S0),
                     ("geometric Asian", G, geo_exact),
                     ("both", np.column_stack([G, X2]), [geo_exact, S0])]:
    if X is None:
        p, se, _ = summary(Y); se0 = se; b = []
    else:
        Ycv, b = control_variate(Y, X, mX); p, se, _ = summary(Ycv)
    print(f"control {label:16s} price {p:.4f}  s.e. {se:.5f}  ratio {(se0/se)**2:7.1f}"
          f"  b = {np.round(b, 3)}")
