import numpy as np
import matplotlib.pyplot as plt
from mcfin_sp import gbm_paths

S0, mu, sigma, T = 100.0, 0.08, 0.25, 2.0
rng = np.random.default_rng(seed=12)
t, S = gbm_paths(rng, S0, mu, sigma, T, n_steps=500, n_paths=20_000)
q = np.percentile(S, [5, 50, 95], axis=0)
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.plot(t, S[:8].T, lw=0.7)
ax.fill_between(t, q[0], q[2], color="#9bb7d4", alpha=0.35, label="5%-95% band")
ax.plot(t, q[1], "k-", lw=1.2, label="median")
ax.plot(t, S0 * np.exp(mu * t), "k--", lw=1.2, label="mean S0 exp(mu t)")
ax.set_xlabel("t (years)"); ax.set_ylabel("S(t)"); ax.legend(loc="upper left", fontsize=7)
plt.tight_layout()
plt.show()
