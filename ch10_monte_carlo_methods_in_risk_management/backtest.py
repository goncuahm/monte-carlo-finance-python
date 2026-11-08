import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mcfin_risk import kupiec

# Synthetic "history": 2,000 days from a GARCH(1,1) model with Student-t(5) shocks
w, a, b, nu, n = 1e-6, 0.08, 0.90, 5, 2000
rng = np.random.default_rng(seed=6)
r, s2 = np.zeros(n), np.zeros(n)
s2[0] = w / (1 - a - b)
for t in range(n):
    if t > 0:
        s2[t] = w + a * r[t - 1]**2 + b * s2[t - 1]
    r[t] = np.sqrt(s2[t]) * rng.standard_t(nu) / np.sqrt(nu / (nu - 2))
loss = -r
win, alpha = 250, 0.99
t_test = np.arange(win, n)
models = {
    "historical simulation": np.array([np.quantile(loss[t - win:t], alpha) for t in t_test]),
    "normal, 250-day vol": np.array([norm.ppf(alpha) * r[t - win:t].std() for t in t_test]),
    "normal, GARCH vol": norm.ppf(alpha) * np.sqrt(s2[t_test]),
}
print(f"test days {t_test.size}, expected exceptions {t_test.size * (1 - alpha):.1f}\n")
print("model                    exceptions   Kupiec LR   p-value")
for k, v in models.items():
    x = int(np.sum(loss[t_test] > v))
    lr, pv = kupiec(x, t_test.size, 1 - alpha)
    print(f"{k:24s} {x:8d} {lr:12.2f} {pv:9.3f}")
fig, ax = plt.subplots(figsize=(6.6, 2.8))
ax.plot(t_test, 100 * loss[t_test], lw=0.4, color="0.6", label="daily loss")
ax.plot(t_test, 100 * models["historical simulation"], lw=1, label="99% VaR, historical")
ax.plot(t_test, 100 * models["normal, GARCH vol"], lw=0.8, label="99% VaR, GARCH")
hit = loss[t_test] > models["historical simulation"]
ax.plot(t_test[hit], 100 * loss[t_test][hit], "x", ms=3, color="#b5523b")
ax.set_ylim(-4, 4.5)
ax.set_xlabel("day"); ax.set_ylabel("loss (%)"); ax.legend(fontsize=7, ncol=3, loc="upper left")
plt.tight_layout()
plt.show()
