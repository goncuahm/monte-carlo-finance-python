import numpy as np
from mcfin_qmc import path_matrix

# Share of the variance of the path average (1/n) sum W(t_i) carried by the first k normals
n, T = 64, 1.0
print("construction   k=1      k=2      k=4      k=8     k=16")
for method in ["standard", "bridge", "pca"]:
    A = path_matrix(n, T, method)
    c = A.mean(axis=0)                     # average = c @ Z, Var = sum c_k^2
    share = np.cumsum(c**2) / np.sum(c**2)
    cov_ok = np.allclose(A @ A.T, np.minimum.outer(*(2 * [np.arange(1, n + 1) / n])))
    print(f"{method:10s} " + " ".join(f"{share[k-1]:8.4f}" for k in [1, 2, 4, 8, 16])
          + f"   cov OK: {cov_ok}")
