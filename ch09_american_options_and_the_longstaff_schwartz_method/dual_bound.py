import numpy as np
from mcfin_amer import bs_put, gbm_paths, lsm, apply_policy

# Lower bound: LSM policy on new paths.  Upper bound (Rogers 2002): for any
# martingale M with M_0 = 0,  V_0 <= E[ max_j ( e^{-r t_j} h(S_j) - M_j ) ].
# We use a multiple of the discounted European put price process as M.
S0, K, r, sigma, T, M = 36.0, 40.0, 0.06, 0.2, 1.0, 50
dt, t = T / M, np.linspace(0, T, M + 1)
put = lambda s: np.maximum(K - s, 0)
_, coefs = lsm(gbm_paths(np.random.default_rng(1), S0, r, sigma, T, M, 100_000),
               put, r, dt, scale=K)
S = gbm_paths(np.random.default_rng(3), S0, r, sigma, T, M, 100_000)
v = apply_policy(S, put, r, dt, coefs, scale=K)
lo, se_lo = v.mean(), v.std() / np.sqrt(v.size)

E = np.exp(-r * t) * bs_put(S, K, r, sigma, T - t)           # martingale (discounted)
H = np.exp(-r * t) * put(S)                                  # discounted exercise value
def upper(beta, S_idx):
    D = np.max(H[S_idx] - beta * (E[S_idx] - E[0, 0]), axis=1)
    return D.mean(), D.std() / np.sqrt(D.size)
pilot, test = np.arange(20_000), np.arange(20_000, 100_000)
betas = np.linspace(0.8, 1.6, 33)
best = betas[np.argmin([upper(b, pilot)[0] for b in betas])]  # choose on pilot paths
hi, se_hi = upper(best, test)                                 # evaluate on fresh paths
print(f"lower bound (LSM policy)   {lo:.4f}  s.e. {se_lo:.4f}")
print(f"upper bound (beta = {best:.3f}) {hi:.4f}  s.e. {se_hi:.4f}")
print(f"95% interval for the Bermudan price [{lo - 1.96*se_lo:.4f}, {hi + 1.96*se_hi:.4f}]")
