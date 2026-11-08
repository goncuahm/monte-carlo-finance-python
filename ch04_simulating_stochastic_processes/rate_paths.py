import numpy as np
import matplotlib.pyplot as plt

kappa, theta, r0, T, n = 0.5, 0.04, 0.02, 10.0, 1000
dt = T / n
t = np.linspace(0, T, n + 1)
rng = np.random.default_rng(seed=30)
Z = rng.standard_normal((3, n))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
for title, sig, model, ax in [("Vasicek, sigma = 0.02", 0.02, "vas", axs[0]),
                              ("CIR, sigma = 0.15", 0.15, "cir", axs[1])]:
    for p in range(3):
        r = np.empty(n + 1); r[0] = r0
        for k in range(n):
            vol = sig if model == "vas" else sig * np.sqrt(max(r[k], 0))
            r[k + 1] = r[k] + kappa * (theta - r[k]) * dt + vol * np.sqrt(dt) * Z[p, k]
        ax.plot(t, 100 * r, lw=0.8)
    ax.axhline(100 * theta, color="k", ls="--", lw=0.8); ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(title, fontsize=9); ax.set_xlabel("t (years)")
axs[0].set_ylabel("short rate (%)")
plt.tight_layout()
plt.show()
