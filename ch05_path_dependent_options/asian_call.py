import numpy as np
from mcfin_path import mc_price, gbm_steps, bs_call, geometric_asian_call

S0, K, r, sigma, T, n_paths = 100.0, 100.0, 0.05, 0.30, 1.0, 400_000
print(f"European call (Black-Scholes): {bs_call(S0, K, r, sigma, T):.4f}\n")
print(" n   arithmetic   s.e.   geometric   s.e.   geo. formula  corr")
for n in [4, 12, 52, 252]:
    rng = np.random.default_rng(seed=1)
    s_sum, log_sum = np.zeros(n_paths), np.zeros(n_paths)
    for S in gbm_steps(rng, S0, r, sigma, T, n, n_paths):
        s_sum += S
        log_sum += np.log(S)
    A, G = s_sum / n, np.exp(log_sum / n)                # arithmetic, geometric
    pa = np.exp(-r * T) * np.maximum(A - K, 0)
    pg = np.exp(-r * T) * np.maximum(G - K, 0)
    (a, sa), (g, sg) = mc_price(pa), mc_price(pg)
    print(f"{n:3d} {a:10.4f} {sa:7.4f} {g:10.4f} {sg:7.4f} "
          f"{geometric_asian_call(S0, K, r, sigma, T, n):11.4f} {np.corrcoef(pa, pg)[0, 1]:7.4f}")
