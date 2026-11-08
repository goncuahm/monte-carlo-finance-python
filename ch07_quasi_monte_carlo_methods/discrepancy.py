import numpy as np
from scipy.stats import qmc
from mcfin_qmc import halton

# Centred L2 discrepancy (smaller = more uniform), averaged over a few seeds for random sets
print("  d       N     random     Halton      Sobol")
for d in [2, 8]:
    for m in [6, 8, 10, 12]:
        n = 2**m
        rnd = np.mean([qmc.discrepancy(np.random.default_rng(s).random((n, d)))
                       for s in range(5)])
        hal = qmc.discrepancy(halton(n, d))
        sob = qmc.discrepancy(qmc.Sobol(d, scramble=True, seed=0).random_base2(m))
        print(f"{d:3d} {n:7d} {rnd:10.2e} {hal:10.2e} {sob:10.2e}")
