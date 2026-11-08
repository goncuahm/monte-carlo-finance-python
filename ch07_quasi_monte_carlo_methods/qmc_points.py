import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from mcfin_qmc import halton

n = 256
sets = [("pseudo-random", np.random.default_rng(seed=1).random((n, 2))),
        ("Halton, bases 2 and 3", halton(n, 2)),
        ("Sobol (scrambled)", qmc.Sobol(2, scramble=True, seed=1).random_base2(8)),
        ("Halton, bases 89 and 97", halton(n, 25)[:, 23:25])]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1))
for ax, (title, P) in zip(axs, sets):
    ax.scatter(P[:, 0], P[:, 1], s=2, color="#2a6db5")
    ax.set_title(title, fontsize=8); ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
plt.tight_layout()
plt.show()
