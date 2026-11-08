import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 1.0
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
exact = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))

rng = np.random.default_rng(1)
N = 200_000
Z = rng.standard_normal(N)
pay = np.exp(-r * T) * np.maximum(S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z) - K, 0)
n = np.arange(1, N + 1)
running = np.cumsum(pay) / n
se = pay.std() / np.sqrt(n)

fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.plot(n, running, color="#2a5d8f", lw=1, label="running MC estimate")
ax.fill_between(n, exact - 1.96 * se, exact + 1.96 * se, color="#c9962b", alpha=0.3, label="exact +/- 1.96 se")
ax.axhline(exact, color="black", lw=1)
ax.set_xscale("log"); ax.set_ylim(exact - 2, exact + 2)
ax.set_xlabel("number of simulations n"); ax.set_ylabel("call price"); ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
