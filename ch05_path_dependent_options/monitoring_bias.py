import numpy as np
import matplotlib.pyplot as plt
from mcfin_path import down_out_call, geometric_asian_call

S0, K, B, r, sigma, T = 100.0, 100.0, 85.0, 0.05, 0.30, 1.0
n = np.unique(np.round(np.logspace(0.3, 3, 40)).astype(int))
B_shift = B * np.exp(-0.5826 * sigma * np.sqrt(T / n))
barrier = down_out_call(S0, K, B_shift, r, sigma, T) - down_out_call(S0, K, B, r, sigma, T)
asian = geometric_asian_call(S0, K, r, sigma, T, n) - geometric_asian_call(S0, K, r, sigma, T, 10**6)
fig, ax = plt.subplots(figsize=(5.8, 2.9))
ax.semilogx(n, barrier, label="down-and-out call (discrete - continuous)")
ax.semilogx(n, asian, label="geometric Asian (n dates - continuous)")
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("number of monitoring dates n"); ax.set_ylabel("price difference")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
