import numpy as np
import matplotlib.pyplot as plt
from mcfin_sp import merton_paths

rng = np.random.default_rng(seed=11)
t, S = merton_paths(rng, S0=100, r=0.03, sigma=0.15, lam=2.0, muJ=-0.10, delJ=0.08,
                    T=2.0, n_steps=1000, n_paths=5)
fig, ax = plt.subplots(figsize=(6.4, 2.7))
ax.plot(t, S.T, lw=0.8)
ax.set_xlabel("t (years)"); ax.set_ylabel("S(t)")
plt.tight_layout()
plt.show()
