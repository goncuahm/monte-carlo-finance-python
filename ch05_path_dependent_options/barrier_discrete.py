import numpy as np
from mcfin_path import mc_price, gbm_steps, down_out_call

S0, K, B, r, sigma, T, n_paths = 100.0, 100.0, 85.0, 0.05, 0.30, 1.0, 400_000
cont = down_out_call(S0, K, B, r, sigma, T)
print(f"continuously monitored down-and-out call: {cont:.4f}\n")
print("   n   MC (discrete)   s.e.   shifted-barrier formula")
for n in [4, 12, 52, 252, 1000]:
    rng = np.random.default_rng(seed=2)
    alive = np.ones(n_paths, dtype=bool)
    for S in gbm_steps(rng, S0, r, sigma, T, n, n_paths):
        alive &= S > B                                  # knocked out at a monitoring date
    pay = np.exp(-r * T) * np.maximum(S - K, 0) * alive
    p, se = mc_price(pay)
    B_shift = B * np.exp(-0.5826 * sigma * np.sqrt(T / n))   # Broadie-Glasserman-Kou
    print(f"{n:5d} {p:12.4f} {se:9.4f} {down_out_call(S0, K, B_shift, r, sigma, T):14.4f}")
