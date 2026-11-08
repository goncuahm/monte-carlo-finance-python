import numpy as np
from mcfin_rng import MINSTD

g = MINSTD(seed=2024)
print("      n     estimate    error    std. error")
for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
    u = np.array(g.sample(2 * n)).reshape(n, 2)
    inside = (u ** 2).sum(axis=1) <= 1.0          # point in the quarter circle?
    p_hat = inside.mean()
    est = 4 * p_hat
    se = 4 * np.sqrt(p_hat * (1 - p_hat) / n)
    print(f"{n:9,d}   {est:.5f}   {est - np.pi:+.5f}   {se:.5f}")
