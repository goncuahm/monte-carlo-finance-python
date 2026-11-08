import numpy as np
import matplotlib.pyplot as plt
from mcfin_qmc import sobol_normals, path_matrix

# Arithmetic Asian call with 64 averaging dates: RMSE over 16 scramblings (or seeds)
S0, K, r, sigma, T, n, R = 100.0, 100.0, 0.05, 0.3, 1.0, 64, 16
t = T * np.arange(1, n + 1) / n
def price(Z, A):
    W = Z @ A.T
    S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W)
    return np.exp(-r * T) * np.maximum(S.mean(axis=1) - K, 0).mean()

A = {m: path_matrix(n, T, m) for m in ["standard", "bridge", "pca"]}
ref = np.mean([price(sobol_normals(18, n, 1000 + s), A["pca"]) for s in range(8)])
print(f"reference price (8 x 2^18 Sobol-PCA points) {ref:.5f}\n")
ms = [8, 10, 12, 14]
res = {"MC": []} | {f"Sobol {k}": [] for k in A}
for m in ms:
    e = [price(np.random.default_rng(s).standard_normal((2**m, n)), A["standard"]) - ref
         for s in range(R)]
    res["MC"].append(np.sqrt(np.mean(np.square(e))))
    for k in A:
        e = [price(sobol_normals(m, n, s), A[k]) - ref for s in range(R)]
        res[f"Sobol {k}"].append(np.sqrt(np.mean(np.square(e))))
print("     N   " + "".join(f"{k:>16s}" for k in res))
for i, m in enumerate(ms):
    print(f"{2**m:6d}   " + "".join(f"{res[k][i]:16.5f}" for k in res))
fig, ax = plt.subplots(figsize=(5.6, 2.9))
for k, mk in zip(res, ["o-", "s-", "^-", "d-"]):
    ax.loglog(2.0**np.array(ms), res[k], mk, ms=3, label=k)
ax.set_xlabel("number of paths N"); ax.set_ylabel("RMSE of price"); ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
