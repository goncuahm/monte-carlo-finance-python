import numpy as np
from scipy.stats import norm, qmc

# Basket call of Chapter 3/6 (3 assets): plain MC, Latin hypercube, scrambled Sobol
S0, sig, w = np.array([100., 100., 100.]), np.array([0.2, 0.3, 0.25]), np.ones(3) / 3
K, r, T, rho, m, R = 100.0, 0.03, 1.0, 0.5, 14, 32
L = np.linalg.cholesky(np.full((3, 3), rho) + (1 - rho) * np.eye(3))
def price(Z):
    ST = S0 * np.exp((r - 0.5 * sig**2) * T + sig * np.sqrt(T) * (Z @ L.T))
    return np.exp(-r * T) * np.maximum(ST @ w - K, 0).mean()
n = 2**m
plain = [price(np.random.default_rng(s).standard_normal((n, 3))) for s in range(R)]
lhs = [price(norm.ppf(qmc.LatinHypercube(3, seed=s).random(n))) for s in range(R)]
sob = [price(norm.ppf(qmc.Sobol(3, scramble=True, seed=s).random_base2(m))) for s in range(R)]
v0 = np.var(plain, ddof=1)
for name, e in [("plain MC", plain), ("Latin hypercube", lhs), ("scrambled Sobol", sob)]:
    print(f"{name:16s} mean {np.mean(e):.5f}  std.dev. {np.std(e, ddof=1):.5f}"
          f"  variance ratio {v0 / np.var(e, ddof=1):7.1f}")
