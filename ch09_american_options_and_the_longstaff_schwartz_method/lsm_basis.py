import numpy as np
from mcfin_amer import binomial_put, gbm_paths, lsm, apply_policy

S0, K, r, sigma, T, M, N = 36.0, 40.0, 0.06, 0.2, 1.0, 50, 50_000
dt = T / M
put = lambda s: np.maximum(K - s, 0)
S_fit = gbm_paths(np.random.default_rng(1), S0, r, sigma, T, M, N)
S_new = gbm_paths(np.random.default_rng(2), S0, r, sigma, T, M, 200_000)
print(f"reference (binomial) {binomial_put(S0, K, r, sigma, T, 5000):.4f}\n")
print("degree   regression on    out-of-sample price   s.e.")
for itm_only in [True, False]:
    for deg in [1, 2, 3, 5]:
        _, coefs = lsm(S_fit, put, r, dt, degree=deg, scale=K, itm_only=itm_only)
        v = apply_policy(S_new, put, r, dt, coefs, degree=deg, scale=K)
        label = "ITM paths" if itm_only else "all paths"
        print(f"{deg:5d}    {label:14s} {v.mean():12.4f} {v.std()/np.sqrt(v.size):11.4f}")
