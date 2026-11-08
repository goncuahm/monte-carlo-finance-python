import numpy as np
from scipy.stats import norm

# Basket call on three assets (Chapter 3): plain sampling versus Latin hypercube
S0, sig, w = np.array([100., 100., 100.]), np.array([0.2, 0.3, 0.25]), np.ones(3) / 3
K, r, T, rho, N, n_rep = 100.0, 0.03, 1.0, 0.5, 20_000, 200
L = np.linalg.cholesky(np.full((3, 3), rho) + (1 - rho) * np.eye(3))

def price(Z):
    X = Z @ L.T
    ST = S0 * np.exp((r - 0.5 * sig**2) * T + sig * np.sqrt(T) * X)
    return np.exp(-r * T) * np.maximum(ST @ w - K, 0).mean()

def latin_hypercube(rng, N, d):
    # each column: one point in each of N equiprobable strata, randomly permuted
    U = (np.argsort(rng.random((N, d)), axis=0) + rng.random((N, d))) / N
    return norm.ppf(U)

plain = [price(np.random.default_rng(i).standard_normal((N, 3))) for i in range(n_rep)]
lhs = [price(latin_hypercube(np.random.default_rng(i), N, 3)) for i in range(n_rep)]
print(f"plain MC        mean {np.mean(plain):.4f}  std.dev. {np.std(plain, ddof=1):.4f}")
print(f"Latin hypercube mean {np.mean(lhs):.4f}  std.dev. {np.std(lhs, ddof=1):.4f}")
print(f"variance ratio {np.var(plain, ddof=1) / np.var(lhs, ddof=1):.1f}")
