import numpy as np
from mcfin_amer import binomial_put, gbm_paths, lsm, apply_policy

S0, K, r, sigma, T, M = 36.0, 40.0, 0.06, 0.2, 1.0, 50   # 50 exercise dates per year
dt = T / M
put = lambda s: np.maximum(K - s, 0)
print(f"American put, binomial (5000 steps)  {binomial_put(S0, K, r, sigma, T, 5000):.4f}\n")
print("     N    in-sample     out-of-sample (s.e.)    P(early exercise)")
for N in [1_000, 10_000, 100_000]:
    rng = np.random.default_rng(seed=1)
    price_in, coefs = lsm(gbm_paths(rng, S0, r, sigma, T, M, N), put, r, dt, scale=K)
    S_new = gbm_paths(np.random.default_rng(seed=2), S0, r, sigma, T, M, 200_000)
    v = apply_policy(S_new, put, r, dt, coefs, scale=K)
    early = np.mean(put(S_new[:, -1]) * np.exp(-r * T) != v)
    print(f"{N:7,d} {price_in:11.4f} {v.mean():12.4f} ({v.std()/np.sqrt(v.size):.4f})"
          f" {early:14.3f}")
