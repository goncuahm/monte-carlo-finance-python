import numpy as np
from mcfin_rng import MINSTD

u = np.array(MINSTD(seed=12345).sample(100_000))
print(f"n = {u.size}")
print(f"min = {u.min():.6f}, max = {u.max():.6f}")
print(f"mean     = {u.mean():.5f}   (theory 1/2  = 0.50000)")
print(f"variance = {u.var(ddof=1):.5f}   (theory 1/12 = {1/12:.5f})")
print(f"P(U < 0.25) estimate = {(u < 0.25).mean():.5f}   (theory 0.25)")
