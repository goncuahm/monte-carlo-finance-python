import numpy as np
from mcfin_path import mc_price, gbm_steps, bs_call

S0, K, r, sigma, T, n, n_paths = 100.0, 100.0, 0.05, 0.30, 1.0, 252, 400_000
print(" barrier  down-out  down-in   sum     vanilla  P(knock-out)")
for B in [70, 80, 90, 95]:
    rng = np.random.default_rng(seed=4)
    alive = np.ones(n_paths, dtype=bool)
    for S in gbm_steps(rng, S0, r, sigma, T, n, n_paths):
        alive &= S > B
    call = np.exp(-r * T) * np.maximum(S - K, 0)
    out, inn = mc_price(call * alive)[0], mc_price(call * ~alive)[0]
    print(f"{B:6d} {out:10.4f} {inn:8.4f} {out + inn:8.4f} {bs_call(S0, K, r, sigma, T):8.4f}"
          f" {1 - alive.mean():10.4f}")
