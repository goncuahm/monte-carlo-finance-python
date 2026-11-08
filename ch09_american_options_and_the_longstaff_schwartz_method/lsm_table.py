import numpy as np
from mcfin_amer import binomial_put, bs_put, gbm_paths, lsm, apply_policy

K, r, M_per_year, N = 40.0, 0.06, 50, 100_000
put = lambda s: np.maximum(K - s, 0)
print("  S0  sigma  T   binomial   LSM (s.e.)        European   premium")
for S0 in [36.0, 40.0, 44.0]:
    for sigma in [0.2, 0.4]:
        for T in [1.0, 2.0]:
            M = int(M_per_year * T); dt = T / M
            _, coefs = lsm(gbm_paths(np.random.default_rng(1), S0, r, sigma, T, M, N),
                           put, r, dt, scale=K)
            v = apply_policy(gbm_paths(np.random.default_rng(2), S0, r, sigma, T, M, N),
                             put, r, dt, coefs, scale=K)
            b = binomial_put(S0, K, r, sigma, T, 2000)
            e = bs_put(S0, K, r, sigma, T)
            print(f"{S0:4.0f}  {sigma:4.1f}  {T:.0f} {b:9.3f} {v.mean():8.3f} "
                  f"({v.std()/np.sqrt(N):.3f}) {e:10.3f} {b - e:8.3f}")
