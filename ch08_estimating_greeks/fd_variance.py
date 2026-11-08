import numpy as np
import matplotlib.pyplot as plt
from mcfin_greeks import terminal

S0, K, r, sigma, T, N = 100.0, 100.0, 0.05, 0.2, 1.0, 100_000
call = lambda s0, Z: np.exp(-r * T) * np.maximum(terminal(s0, r, sigma, T, Z) - K, 0)
hs = np.logspace(-3, 1, 17)
rng = np.random.default_rng(seed=2)
Z1, Z2, Z3 = (rng.standard_normal(N) for _ in range(3))
se = {"delta, independent": [], "delta, CRN": [], "gamma, CRN": []}
for h in hs:
    se["delta, independent"].append(np.std((call(S0 + h, Z1) - call(S0 - h, Z2)) / (2*h)))
    se["delta, CRN"].append(np.std((call(S0 + h, Z1) - call(S0 - h, Z1)) / (2 * h)))
    g = (call(S0 + h, Z1) - 2 * call(S0, Z1) + call(S0 - h, Z1)) / h**2
    se["gamma, CRN"].append(np.std(g))
fig, ax = plt.subplots(figsize=(5.6, 2.9))
for k, mk in zip(se, ["o-", "s-", "^-"]):
    ax.loglog(hs, np.array(se[k]) / np.sqrt(N), mk, ms=3, label=k)
ax.set_xlabel("bump size h"); ax.set_ylabel("standard error"); ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
