import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Test integral over [0,1]^d with exact value 1 (Sobol' g-function, a_j = j)
d = 8
a = np.arange(1, d + 1)
f = lambda U: np.prod((np.abs(4 * U - 2) + a) / (1 + a), axis=1)
ms, n_rep = range(6, 17), 20
rmse = {"Monte Carlo": [], "scrambled Halton": [], "scrambled Sobol": []}
for m in ms:
    n = 2**m
    e_mc = [f(np.random.default_rng(s).random((n, d))).mean() - 1 for s in range(n_rep)]
    e_h = [f(qmc.Halton(d, scramble=True, seed=s).random(n)).mean() - 1 for s in range(n_rep)]
    e_s = [f(qmc.Sobol(d, scramble=True, seed=s).random_base2(m)).mean() - 1
           for s in range(n_rep)]
    for key, e in zip(rmse, [e_mc, e_h, e_s]):
        rmse[key].append(np.sqrt(np.mean(np.square(e))))
N = 2.0**np.array(ms)
print("      N     MC rmse    Halton rmse   Sobol rmse")
for i in range(0, len(N), 2):
    print(f"{int(N[i]):7d} {rmse['Monte Carlo'][i]:10.2e} {rmse['scrambled Halton'][i]:12.2e}"
          f" {rmse['scrambled Sobol'][i]:12.2e}")
for key in rmse:
    print(f"slope of log rmse against log N, {key:16s}: "
          f"{np.polyfit(np.log(N), np.log(rmse[key]), 1)[0]:.2f}")
fig, ax = plt.subplots(figsize=(5.6, 2.9))
for key, mk in zip(rmse, ["o-", "s-", "^-"]):
    ax.loglog(N, rmse[key], mk, ms=3, label=key)
ax.set_xlabel("number of points N"); ax.set_ylabel("root mean square error")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
