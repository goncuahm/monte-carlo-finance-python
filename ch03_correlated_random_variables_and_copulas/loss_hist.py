import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

n_firms, p, n_sims = 100, 0.02, 200_000
c = norm.ppf(p)
rng = np.random.default_rng(17)
fig, ax = plt.subplots(figsize=(6.4, 2.9))
for rho, col in [(0.0, "#6b8fb5"), (0.3, "#b5523b")]:
    M = rng.standard_normal((n_sims, 1))
    d = (np.sqrt(rho) * M + np.sqrt(1 - rho) * rng.standard_normal((n_sims, n_firms)) < c).sum(axis=1)
    freq = np.bincount(d, minlength=31)[:31] / n_sims
    ax.bar(np.arange(31) + (0.2 if rho else -0.2), freq, width=0.4, color=col, label=f"rho = {rho}")
ax.plot(np.arange(31), binom.pmf(np.arange(31), n_firms, p), "k.", ms=4, label="binomial (independent)")
ax.set_yscale("log"); ax.set_ylim(1e-5, 1); ax.set_xlim(-1, 30)
ax.set_xlabel("number of defaults out of 100"); ax.set_ylabel("probability"); ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
