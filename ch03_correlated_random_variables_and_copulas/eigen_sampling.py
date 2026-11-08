import numpy as np

# Generate N(0, C) with the eigen-decomposition C = V diag(lam) V^T: X = V sqrt(lam) Z
C = np.array([[1.0, 0.6, 0.3],
              [0.6, 1.0, 0.5],
              [0.3, 0.5, 1.0]])
lam, V = np.linalg.eigh(C)
A = V @ np.diag(np.sqrt(lam))
rng = np.random.default_rng(seed=3)
X = rng.standard_normal((400_000, 3)) @ A.T

np.set_printoptions(precision=4, suppress=True)
print("eigenvalues:", lam, " share of the largest:", (lam[-1] / lam.sum()).round(3))
print("sample correlation:\n", np.corrcoef(X, rowvar=False))
