import numpy as np
import matplotlib.pyplot as plt
from mcfin_sp import brownian_paths

rng = np.random.default_rng(seed=4)
t, W = brownian_paths(rng, T=1.0, n_steps=500, n_paths=10)
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.plot(t, W.T, lw=0.8)
ax.plot(t, 1.96 * np.sqrt(t), "k--", lw=1, label="+/- 1.96 sqrt(t)")
ax.plot(t, -1.96 * np.sqrt(t), "k--", lw=1)
ax.set_xlabel("t"); ax.set_ylabel("W(t)"); ax.legend(loc="upper left", fontsize=7)
plt.tight_layout()
plt.show()
