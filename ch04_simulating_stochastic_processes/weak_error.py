import numpy as np
from mcfin_sp import bs_call

# Euler weak error for a European call under Black-Scholes (r = 0.05, sigma = 0.4).
# The exact solution is driven by the SAME normals, so the error E[pay_E - pay_X]
# is estimated with a much smaller standard error than either price alone.
S0, K, r, sigma, T, n_paths = 100.0, 100.0, 0.05, 0.4, 1.0, 1_000_000
exact = bs_call(S0, K, r, sigma, T)
print(f"Black-Scholes price {exact:.4f}")
print(" steps   Euler price    error   std.err.  error x steps")
for n in [1, 2, 4, 8, 16, 32]:
    rng = np.random.default_rng(seed=5)
    dt = T / n
    SE, SX = np.full(n_paths, S0), np.full(n_paths, S0)
    for k in range(n):
        dW = np.sqrt(dt) * rng.standard_normal(n_paths)
        SE = SE * (1 + r * dt + sigma * dW)                          # Euler
        SX = SX * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)     # exact
    diff = np.exp(-r * T) * (np.maximum(SE - K, 0) - np.maximum(SX - K, 0))
    err, se = diff.mean(), diff.std() / np.sqrt(n_paths)
    print(f"{n:6d} {exact + err:12.4f} {err:+9.4f} {se:9.4f} {err * n:+10.3f}")
