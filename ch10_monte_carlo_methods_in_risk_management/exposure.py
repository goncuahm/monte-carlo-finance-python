import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Counterparty exposure of a 5-year FX forward: buy 1m EUR at the forward rate K = F(0,T)
S0, rd, rf, sigma, T, notional = 1.10, 0.04, 0.02, 0.10, 5.0, 1e6
K = S0 * np.exp((rd - rf) * T)
M, N = 60, 100_000                                      # monthly grid
t = np.linspace(0, T, M + 1)
rng = np.random.default_rng(seed=7)
Z = rng.standard_normal((N, M))
logS = np.log(S0) + np.cumsum((rd - rf - 0.5 * sigma**2) * (T / M)
                              + sigma * np.sqrt(T / M) * Z, axis=1)
S = np.hstack([np.full((N, 1), S0), np.exp(logS)])
V = notional * (S * np.exp(-rf * (T - t)) - K * np.exp(-rd * (T - t)))  # forward value
EE = np.mean(np.maximum(V, 0), axis=0)                  # expected exposure
PFE = np.quantile(np.maximum(V, 0), 0.975, axis=0)      # potential future exposure
def ee_exact(tt):                                       # Garman-Kohlhagen formula
    if tt == 0: return 0.0
    Kt = K * np.exp(-(rd - rf) * (T - tt))
    d1 = (np.log(S0 / Kt) + (rd - rf + 0.5 * sigma**2) * tt) / (sigma * np.sqrt(tt))
    call = (S0 * np.exp(-rf * tt) * norm.cdf(d1)
            - Kt * np.exp(-rd * tt) * norm.cdf(d1 - sigma * np.sqrt(tt)))
    return notional * np.exp(-rf * (T - tt)) * call * np.exp(rd * tt)
print(" t (years)   EE (MC)    EE exact    PFE 97.5%")
for j in [12, 24, 36, 48, 59]:
    print(f"{t[j]:9.1f} {EE[j]:10,.0f} {ee_exact(t[j]):11,.0f} {PFE[j]:12,.0f}")

# CVA = (1 - R) sum_j discounted EE(t_j) * P(default in (t_{j-1}, t_j])
lam, R = 0.02, 0.4
pd = np.exp(-lam * t[:-1]) - np.exp(-lam * t[1:])
cva = (1 - R) * np.sum(np.exp(-rd * t[1:]) * EE[1:] * pd)
print(f"\nCVA with hazard rate {lam:.0%} and recovery {R:.0%}: {cva:,.0f}")
fig, ax = plt.subplots(figsize=(5.8, 2.9))
ax.plot(t, EE / 1e3, label="expected exposure EE")
ax.plot(t, PFE / 1e3, "--", label="PFE (97.5%)")
ax.plot(t, np.quantile(V, 0.025, axis=0) / 1e3, ":", label="2.5% quantile of value")
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("time (years)"); ax.set_ylabel("thousands"); ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
