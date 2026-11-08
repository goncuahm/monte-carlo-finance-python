import numpy as np
from mcfin_vr import bs_call, summary, Timer

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
exact = bs_call(S0, K, r, sigma, T)
print(f"Black-Scholes price {exact:.4f}\n")
print("        N      price    std.err.   time (s)   var x time")
for N in [10_000, 100_000, 1_000_000, 10_000_000]:
    rng = np.random.default_rng(seed=1)
    with Timer() as t:
        Z = rng.standard_normal(N)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        Y = np.exp(-r * T) * np.maximum(ST - K, 0)
    p, se, _ = summary(Y)
    print(f"{N:10,d} {p:9.4f} {se:10.5f} {t.s:10.4f} {se**2 * t.s:12.2e}")
