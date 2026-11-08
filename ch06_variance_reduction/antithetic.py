import numpy as np
from mcfin_vr import summary

S0, K, r, sigma, T, n_pairs = 100.0, 100.0, 0.05, 0.2, 1.0, 500_000
rng = np.random.default_rng(seed=2)
Z = rng.standard_normal(n_pairs)
disc = np.exp(-r * T)
S = lambda z: S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
payoffs = {"call":     lambda s: np.maximum(s - K, 0),
           "put":      lambda s: np.maximum(K - s, 0),
           "digital":  lambda s: (s > K).astype(float),
           "forward":  lambda s: s - K,
           "straddle": lambda s: np.abs(s - K)}

Z2 = rng.standard_normal(2 * n_pairs)               # plain MC with 2 n_pairs draws
print("payoff      price    corr(f(Z),f(-Z))   variance ratio plain/antithetic")
for name, f in payoffs.items():
    plain = disc * f(S(Z2))
    anti = disc * 0.5 * (f(S(Z)) + f(S(-Z)))        # one value per pair
    _, _, v_plain = summary(plain)
    p, _, v_anti = summary(anti, n_payoffs=2 * n_pairs)
    c = np.corrcoef(f(S(Z)), f(S(-Z)))[0, 1]
    print(f"{name:9s} {p:8.4f} {c:12.3f} {v_plain / v_anti:22.2f}")
