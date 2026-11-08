import numpy as np
import matplotlib.pyplot as plt
from mcfin_rv import exponential_inv

lam = 2.0
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
x = np.linspace(0, 3, 300)
ax[0].plot(x, 1 - np.exp(-lam * x), color="black")
for u in [0.2, 0.55, 0.9]:                       # follow three uniforms through F^{-1}
    xu = -np.log(1 - u) / lam
    ax[0].plot([0, xu, xu], [u, u, 0], "--", color="#b5523b", lw=1)
ax[0].set_xlabel("x"); ax[0].set_ylabel("F(x) = u"); ax[0].set_title("inverse transform")
ax[0].set_xlim(0, 3); ax[0].set_ylim(0, 1.02)

sample = exponential_inv(np.random.default_rng(3), lam, 20_000)
ax[1].hist(sample, bins=60, range=(0, 3), density=True, color="#6b8fb5", edgecolor="white")
ax[1].plot(x, lam * np.exp(-lam * x), color="black")
ax[1].set_xlabel("x"); ax[1].set_title("20,000 draws and the density")
plt.tight_layout()
plt.show()
