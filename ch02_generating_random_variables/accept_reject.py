import numpy as np
from mcfin_rv import accept_reject

rng = np.random.default_rng(seed=12)

# (1) Beta(2, 2) density f(x) = 6x(1-x) on [0,1], proposal U(0,1), c = max f = 1.5
f = lambda x: 6 * x * (1 - x)
x, rate = accept_reject(rng, f, lambda r, n: r.random(n), lambda y: np.ones_like(y), 1.5, 100_000)
print(f"Beta(2,2):   mean {x.mean():.4f} (0.5), var {x.var():.4f} (0.05), acceptance {rate:.3f} (1/c = {1/1.5:.3f})")

# (2) half-normal from an exponential(1) proposal, c = sqrt(2e/pi)
c = np.sqrt(2 * np.e / np.pi)
f = lambda y: np.sqrt(2 / np.pi) * np.exp(-y**2 / 2)
g = lambda y: np.exp(-y)
h, rate = accept_reject(rng, f, lambda r, n: r.exponential(1.0, n), g, c, 100_000)
z = h * np.where(rng.random(h.size) < 0.5, -1, 1)    # random sign -> standard normal
print(f"normal:      mean {z.mean():+.4f} (0), var {z.var():.4f} (1), acceptance {rate:.3f} (1/c = {1/c:.3f})")
