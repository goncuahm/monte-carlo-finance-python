import numpy as np

rng = np.random.default_rng(seed=10)
print("integers(1, 7, 10):          ", rng.integers(1, 7, size=10))        # 1..6, high excluded
print("integers(1, 6, 10, endpoint):", rng.integers(1, 6, size=10, endpoint=True))
print("choice of 5 of 0..251, no repeats:", np.sort(rng.choice(252, size=5, replace=False)))
print("permutation of 8 assets:     ", rng.permutation(8))

import random
random.seed(10)
print("random.randint(1, 6):        ", [random.randint(1, 6) for _ in range(10)])  # both ends included
