import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Relative error per sample of the IS estimator of P(Z > c) as a function of mu:
# second moment E_mu[w^2 1{Y > c}] = exp(mu^2) * P(Z > c + mu)
mu = np.linspace(0, 7, 400)
fig, ax = plt.subplots(figsize=(5.8, 2.9))
for c in [3, 4, 5]:
    p = norm.sf(c)
    m2 = np.exp(mu**2) * norm.sf(c + mu)
    rel = np.sqrt(np.maximum(m2 - p**2, 1e-300)) / p
    ax.semilogy(mu, rel, label=f"c = {c}")
    k = np.argmin(rel)
    ax.plot(mu[k], rel[k], "ko", ms=3)
ax.set_xlabel("shift mu"); ax.set_ylabel("relative error per sample")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
