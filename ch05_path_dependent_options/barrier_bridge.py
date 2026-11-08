import numpy as np
from mcfin_path import mc_price, down_out_call

# Continuous barrier with a coarse grid: multiply by the probability that the
# Brownian bridge between two grid points stays above the barrier.
S0, K, B, r, sigma, T, n_paths = 100.0, 100.0, 85.0, 0.05, 0.30, 1.0, 400_000
b = np.log(B)
print(f"continuous formula {down_out_call(S0, K, B, r, sigma, T):.4f}\n")
print("   n   grid only    s.e.    with bridge   s.e.")
for n in [4, 12, 52]:
    rng = np.random.default_rng(seed=3)
    h = T / n
    x = np.full(n_paths, np.log(S0))
    alive, surv = np.ones(n_paths, dtype=bool), np.ones(n_paths)
    for _ in range(n):
        x_new = x + (r - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * rng.standard_normal(n_paths)
        alive &= x_new > b
        p_cross = np.exp(-2 * (x - b) * (x_new - b) / (sigma**2 * h))
        surv *= np.where(alive, 1 - p_cross, 0.0)       # P(no crossing in between)
        x = x_new
    call = np.exp(-r * T) * np.maximum(np.exp(x) - K, 0)
    (p1, s1), (p2, s2) = mc_price(call * alive), mc_price(call * surv)
    print(f"{n:5d} {p1:10.4f} {s1:8.4f} {p2:12.4f} {s2:8.4f}")
