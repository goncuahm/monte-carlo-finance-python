import numpy as np
import matplotlib.pyplot as plt
from mcfin_sp import merton_call, implied_vol

S0, r, sigma, lam, muJ, delJ = 100.0, 0.03, 0.15, 2.0, -0.10, 0.08
K = np.linspace(70, 140, 36)
fig, ax = plt.subplots(figsize=(5.8, 2.9))
for T in [0.1, 0.5, 2.0]:
    iv = [implied_vol(merton_call(S0, k, r, sigma, lam, muJ, delJ, T), S0, k, r, T)
          for k in K]
    ax.plot(K, 100 * np.array(iv), label=f"T = {T}")
ax.axhline(100 * sigma, color="gray", ls="--", lw=0.8, label="diffusion vol 15%")
ax.set_xlabel("strike K"); ax.set_ylabel("implied volatility (%)"); ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
