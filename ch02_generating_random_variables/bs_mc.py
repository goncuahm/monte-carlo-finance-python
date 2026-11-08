import numpy as np
from scipy.stats import norm

def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def mc_call(rng, S0, K, r, sigma, T, n):
    Z = rng.standard_normal(n)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    disc_payoff = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    return disc_payoff.mean(), disc_payoff.std(ddof=1) / np.sqrt(n)

S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 1.0
exact = bs_call(S0, K, r, sigma, T)
rng = np.random.default_rng(seed=2026)
print(f"Black-Scholes price {exact:.4f}")
print("        n    MC price   std.error   95% interval          covers?")
for n in [1_000, 10_000, 100_000, 1_000_000]:
    price, se = mc_call(rng, S0, K, r, sigma, T, n)
    lo, hi = price - 1.96 * se, price + 1.96 * se
    print(f"{n:9,d}   {price:8.4f}   {se:8.4f}    [{lo:7.4f}, {hi:7.4f}]   {lo <= exact <= hi}")
