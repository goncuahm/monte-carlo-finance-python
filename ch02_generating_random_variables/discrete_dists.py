import numpy as np
from math import comb, exp, factorial
from mcfin_rv import discrete_inv

rng = np.random.default_rng(seed=7)
n = 200_000

# Bernoulli(p): X = 1 if U < p
p = 0.3
bern = (rng.random(n) < p).astype(int)
print(f"Bernoulli(0.3): mean {bern.mean():.4f} (theory 0.3000)")

# Binomial(10, 0.3) by inversion of its pmf
pmf = [comb(10, j) * p**j * (1 - p)**(10 - j) for j in range(11)]
binom = discrete_inv(rng, np.arange(11), pmf, n)
print(f"Binomial(10,0.3): mean {binom.mean():.4f} (theory 3.0000), var {binom.var():.4f} (theory 2.1000)")

# Geometric(p) number of trials until the first success, closed form
geo = np.ceil(np.log1p(-rng.random(n)) / np.log(1 - p)).astype(int)
print(f"Geometric(0.3): mean {geo.mean():.4f} (theory 1/p = {1/p:.4f})")

# Poisson(4) by inversion (truncated at 30, where the tail is negligible)
mu = 4.0
pmf = [exp(-mu) * mu**j / factorial(j) for j in range(31)]
pois = discrete_inv(rng, np.arange(31), pmf, n)
freq = np.bincount(pois, minlength=8)[:8] / n
print("Poisson(4)  k :    " + "  ".join(f"{k:5d}" for k in range(8)))
print("  simulated     " + "  ".join(f"{f:.3f}" for f in freq))
print("  exact pmf     " + "  ".join(f"{q:.3f}" for q in pmf[:8]))
