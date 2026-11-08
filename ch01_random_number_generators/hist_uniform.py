import numpy as np
import matplotlib.pyplot as plt
from mcfin_rng import MINSTD

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for axis, n in zip(ax, [500, 100_000]):
    u = np.array(MINSTD(seed=7).sample(n))
    axis.hist(u, bins=20, density=True, color="#6b8fb5", edgecolor="white")
    axis.axhline(1.0, color="black", lw=1)
    axis.set_title(f"n = {n:,}")
    axis.set_xlabel("u"); axis.set_ylim(0, 1.6)
ax[0].set_ylabel("density")
plt.tight_layout()
plt.show()
