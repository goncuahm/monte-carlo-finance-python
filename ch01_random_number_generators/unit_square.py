import numpy as np
import matplotlib.pyplot as plt
from mcfin_rng import lcg_uniform, MINSTD

small = np.array(lcg_uniform(seed=1, a=21, c=1, m=256, n=256))    # full period, m = 256
good = np.array(MINSTD(seed=3).sample(2_000))

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5))
ax[0].scatter(small[:-1], small[1:], s=6, color="#b5523b")
ax[0].set_title("LCG a = 21, c = 1, m = 256")
ax[1].scatter(good[:-1:2], good[1::2], s=3, color="#2a5d8f")
ax[1].set_title("MINSTD, 1,000 points")
for axis in ax:
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.set_aspect("equal")
    axis.set_xlabel(r"$u_k$"); axis.set_ylabel(r"$u_{k+1}$")
plt.tight_layout()
plt.show()
