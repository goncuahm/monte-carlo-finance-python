import numpy as np
from mcfin_amer import binomial_put, bs_put

S0, K, r, sigma, T = 36.0, 40.0, 0.06, 0.2, 1.0
print(f"European put (Black-Scholes) {bs_put(S0, K, r, sigma, T):.4f}\n")
print("    steps   European tree   American tree   early exercise premium")
for n in [50, 200, 1000, 5000]:
    e = binomial_put(S0, K, r, sigma, T, n, american=False)
    a = binomial_put(S0, K, r, sigma, T, n)
    print(f"{n:9d} {e:15.4f} {a:15.4f} {a - e:15.4f}")
