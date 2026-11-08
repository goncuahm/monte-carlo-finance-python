import numpy as np

# Poisson process with intensity lam: inter-arrival times are exponential(lam).
# The number of events in [0, T] is Poisson(lam * T).
rng = np.random.default_rng(seed=21)
lam, T, n_paths = 3.0, 1.0, 100_000

counts = np.zeros(n_paths, dtype=int)
t = rng.exponential(1 / lam, n_paths)            # first arrival of every path
alive = t <= T
while alive.any():
    counts[alive] += 1
    t[alive] += rng.exponential(1 / lam, alive.sum())
    alive = t <= T

print(f"mean count {counts.mean():.4f}, variance {counts.var():.4f} (both = lam T = {lam*T})")
print(f"P(N = 0)   {np.mean(counts == 0):.4f}  (theory e^-3 = {np.exp(-3):.4f})")
print(f"numpy rng.poisson mean {rng.poisson(lam * T, n_paths).mean():.4f}")
