import numpy as np

rng = np.random.default_rng(seed=1)
n, rho = 500_000, 0.7
Z1, Z2 = rng.standard_normal(n), rng.standard_normal(n)
X1 = Z1
X2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2          # the 2 x 2 Cholesky factor

print(f"var(X1) {X1.var():.4f}, var(X2) {X2.var():.4f}  (both 1)")
print(f"corr(X1, X2) {np.corrcoef(X1, X2)[0, 1]:.4f}  (target {rho})")
print(f"P(X1 > 1 and X2 > 1)  MC {np.mean((X1 > 1) & (X2 > 1)):.4f}"
      f"   independent case {(np.mean(Z1 > 1))**2:.4f}")
