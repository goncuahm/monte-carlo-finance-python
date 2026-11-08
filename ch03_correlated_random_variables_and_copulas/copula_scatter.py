import numpy as np
import matplotlib.pyplot as plt
from mcfin_mv import gaussian_copula, t_copula

corr = np.array([[1.0, 0.7], [0.7, 1.0]])
G = gaussian_copula(np.random.default_rng(8), corr, 3_000)
T = t_copula(np.random.default_rng(8), corr, 3, 3_000)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
for axis, U, title in [(ax[0], G, "Gaussian copula, rho = 0.7"), (ax[1], T, "t copula, rho = 0.7, nu = 3")]:
    axis.scatter(U[:, 0], U[:, 1], s=2, color="#2a5d8f")
    axis.add_patch(plt.Rectangle((0.95, 0.95), 0.05, 0.05, fill=False, color="#b5523b", lw=1.5))
    axis.add_patch(plt.Rectangle((0, 0), 0.05, 0.05, fill=False, color="#b5523b", lw=1.5))
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.set_aspect("equal"); axis.set_title(title)
    axis.set_xlabel("U1"); axis.set_ylabel("U2")
plt.tight_layout()
plt.show()
