import numpy as np
from scipy.stats import poisson
from mcfin_sp import poisson_arrivals

lam, T = 3.0, 2.0                       # 3 events per year, 2 years
rng = np.random.default_rng(seed=6)
print("one path, jump times:", np.round(poisson_arrivals(rng, lam, T), 3))

counts = np.array([len(poisson_arrivals(rng, lam, T)) for _ in range(100_000)])
print(f"\nmean N(T) {counts.mean():.4f}  var N(T) {counts.var():.4f}  (theory {lam*T})")
print(" k   simulated  Poisson pmf")
for k in range(0, 13, 2):
    print(f"{k:2d} {np.mean(counts == k):10.4f} {poisson.pmf(k, lam * T):11.4f}")
