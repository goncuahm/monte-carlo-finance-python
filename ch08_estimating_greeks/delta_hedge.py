import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mcfin_greeks import bs_greeks

# Sell one call at the Black-Scholes price and delta-hedge it n times until maturity.
S0, K, r, sigma, mu, T, N = 100.0, 100.0, 0.05, 0.2, 0.10, 0.25, 50_000
price0 = bs_greeks(S0, K, r, sigma, T)["price"]
def delta(S, tau):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)
results = {}
print(f"call price {price0:.4f}; real-world drift mu = {mu}\n")
print("  rebalances   mean P&L   std P&L   std/price   5% quantile")
for n in [4, 13, 52, 252]:
    rng = np.random.default_rng(seed=9)
    h = T / n
    S = np.full(N, S0)
    d = delta(S, T)
    cash = price0 - d * S                       # premium received minus shares bought
    for k in range(1, n + 1):
        S = S * np.exp((mu - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * rng.standard_normal(N))
        cash *= np.exp(r * h)
        if k < n:
            d_new = delta(S, T - k * h)
            cash -= (d_new - d) * S             # rebalance the share position
            d = d_new
    pnl = cash + d * S - np.maximum(S - K, 0)   # close the hedge, pay the option
    results[n] = pnl
    print(f"{n:10d} {pnl.mean():10.4f} {pnl.std():9.4f} {pnl.std() / price0:10.3f}"
          f" {np.quantile(pnl, 0.05):11.4f}")
fig, ax = plt.subplots(figsize=(5.8, 2.9))
for n in [13, 52, 252]:
    ax.hist(results[n], bins=120, range=(-3, 3), histtype="step", density=True,
            label=f"{n} rebalances")
ax.set_xlabel("hedging profit and loss at maturity"); ax.set_ylabel("density")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
