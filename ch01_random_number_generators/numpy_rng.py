import random
import numpy as np

# Python's built-in module: Mersenne Twister (MT19937)
random.seed(2026)
print("random.random():", [round(random.random(), 4) for _ in range(3)])

# NumPy's recommended interface: a Generator with the PCG64 bit generator
rng = np.random.default_rng(seed=2026)
print("rng.random(3):  ", rng.random(3).round(4))
print("bit generator:  ", rng.bit_generator.__class__.__name__)

# same seed -> same numbers (reproducibility)
a = np.random.default_rng(7).random(4)
b = np.random.default_rng(7).random(4)
print("reproducible:   ", np.array_equal(a, b))

# independent streams for parallel simulations
children = np.random.SeedSequence(2026).spawn(3)
streams = [np.random.default_rng(s) for s in children]
print("three streams:  ", [round(g.random(), 4) for g in streams])
