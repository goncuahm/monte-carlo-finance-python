import numpy as np
import matplotlib.pyplot as plt
from mcfin_risk import bs_value, bs_delta_gamma, var_es

# Book on one stock: short 1,000 ATM straddles (3 months), long 1,000 calls K = 110
S0, r, sigma, h, N = 100.0, 0.03, 0.25, 10 / 252, 200_000   # 10-day horizon
book = [(-1000, 100.0, 0.25, "call"), (-1000, 100.0, 0.25, "put"),
        (+1000, 110.0, 0.25, "call")]
def value(S, tau_shift=0.0):
    return sum(q * bs_value(S, K, r, sigma, T - tau_shift, k) for q, K, T, k in book)
V0 = value(np.array(S0))
delta = sum(q * bs_delta_gamma(S0, K, r, sigma, T, k)[0] for q, K, T, k in book)
gamma = sum(q * bs_delta_gamma(S0, K, r, sigma, T, k)[1] for q, K, T, k in book)
theta = value(np.array(S0), h) - V0                     # value change from time alone
print(f"book value {V0:,.0f}   delta {delta:,.1f}   gamma {gamma:,.2f}   10-day theta {theta:,.0f}")

rng = np.random.default_rng(seed=4)
S_h = S0 * np.exp((r - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * rng.standard_normal(N))
dS = S_h - S0
pnl = {"full revaluation": value(S_h, h) - V0,
       "delta": delta * dS + theta,
       "delta-gamma": delta * dS + 0.5 * gamma * dS**2 + theta}
print("\nmethod              99% VaR     99% ES     97.5% ES")
for k, p in pnl.items():
    v, e = var_es(-p, 0.99); _, e975 = var_es(-p, 0.975)
    print(f"{k:18s} {v:10,.0f} {e:10,.0f} {e975:10,.0f}")
fig, ax = plt.subplots(figsize=(5.8, 2.9))
for k, ls in zip(pnl, ["-", "--", ":"]):
    ax.hist(pnl[k] / 1000, bins=140, range=(-12, 2), density=True, histtype="step",
            ls=ls, lw=1.2, label=k)
ax.set_yscale("log")
ax.set_xlabel("10-day profit and loss (thousands)"); ax.set_ylabel("density (log scale)")
ax.legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.show()
