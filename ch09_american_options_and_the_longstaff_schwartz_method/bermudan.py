import numpy as np
from mcfin_amer import binomial_put, bs_put, gbm_paths, lsm, apply_policy

S0, K, r, sigma, T, N = 36.0, 40.0, 0.06, 0.2, 1.0, 100_000
put = lambda s: np.maximum(K - s, 0)
print(f"European {bs_put(S0, K, r, sigma, T):.4f}   American (binomial) "
      f"{binomial_put(S0, K, r, sigma, T, 5000):.4f}\n")
print(" exercise dates   LSM price (s.e.)")
for M in [1, 2, 4, 12, 50, 250]:
    dt = T / M
    _, coefs = lsm(gbm_paths(np.random.default_rng(1), S0, r, sigma, T, M, N),
                   put, r, dt, scale=K)
    v = apply_policy(gbm_paths(np.random.default_rng(2), S0, r, sigma, T, M, N),
                     put, r, dt, coefs, scale=K)
    print(f"{M:10d} {v.mean():14.4f} ({v.std()/np.sqrt(N):.4f})")
