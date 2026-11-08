import numpy as np
import matplotlib.pyplot as plt

S0, B, r, sigma, T, n = 100.0, 85.0, 0.05, 0.30, 1.0, 252
rng = np.random.default_rng(seed=7)
t = np.linspace(0, T, n + 1)
Z = rng.standard_normal((12, n))
S = S0 * np.exp(np.cumsum((r - sigma**2 / 2) * T / n + sigma * np.sqrt(T / n) * Z, axis=1))
S = np.hstack([np.full((12, 1), S0), S])
fig, ax = plt.subplots(figsize=(6.4, 2.8))
for path in S:
    hit = np.argmax(path <= B) if (path <= B).any() else None
    if hit is None:
        ax.plot(t, path, lw=0.9, ls="-", color="#163f73")
    else:
        ax.plot(t[:hit + 1], path[:hit + 1], lw=0.8, ls="--", color="#d9734e")
        ax.plot(t[hit:], path[hit:], lw=0.6, ls="-", color="#d9734e", alpha=0.25)
        ax.plot(t[hit], path[hit], "x", color="#d9734e")
ax.axhline(B, color="k", ls="--", lw=1)
ax.text(0.80, B + 3, "barrier B = 85", fontsize=8)
ax.set_xlabel("t (years)"); ax.set_ylabel("S(t)")
plt.tight_layout()
plt.show()
