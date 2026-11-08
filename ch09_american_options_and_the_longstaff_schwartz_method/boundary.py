import numpy as np
import matplotlib.pyplot as plt
from mcfin_amer import binomial_put, gbm_paths, lsm, basis

S0, K, r, sigma, T, M, N = 36.0, 40.0, 0.06, 0.2, 1.0, 50, 100_000
dt = T / M
put = lambda s: np.maximum(K - s, 0)
_, bnd = binomial_put(S0, K, r, sigma, T, 2500, boundary=True)
_, coefs = lsm(gbm_paths(np.random.default_rng(1), S0, r, sigma, T, M, N), put, r, dt,
               scale=K)
grid = np.linspace(20, 40, 2001)
t_lsm, b_lsm = [], []
for j in range(1, M):
    if coefs[j] is None:
        continue
    gain = put(grid) - basis(grid / K, 3) @ coefs[j]      # exercise minus continuation
    ex = np.where(gain > 0)[0]
    if ex.size:
        t_lsm.append(j * dt); b_lsm.append(grid[ex.max()])
fig, ax = plt.subplots(figsize=(5.8, 2.9))
ax.plot(np.linspace(0, T, bnd.size), bnd, "k-", lw=1.2, label="binomial (2500 steps)")
ax.plot(t_lsm, b_lsm, "o", ms=3, color="#b5523b", label="LSM, 100,000 paths")
ax.set_ylim(28, 41); ax.set_xlabel("time t"); ax.set_ylabel("critical stock price")
ax.text(0.05, 29, "exercise region", fontsize=8); ax.text(0.05, 39, "continuation region", fontsize=8)
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.show()
