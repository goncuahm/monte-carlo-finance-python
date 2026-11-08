import numpy as np

# Annual operational losses: N ~ Poisson(lam) events, sizes lognormal(m, s)
lam, m, s, n_years = 10.0, 11.0, 1.5, 200_000
rng = np.random.default_rng(seed=19)
N = rng.poisson(lam, n_years)
sizes = rng.lognormal(m, s, N.sum())                 # all losses in one array
year = np.repeat(np.arange(n_years), N)              # year of each loss
L = np.bincount(year, weights=sizes, minlength=n_years)

EX, EX2 = np.exp(m + s**2 / 2), np.exp(2 * m + 2 * s**2)
print(f"mean annual loss {L.mean():12,.0f}   theory lam E[X]      {lam*EX:12,.0f}")
print(f"std annual loss  {L.std():12,.0f}   theory sqrt(lam E[X^2]) {np.sqrt(lam*EX2):9,.0f}")
for q in [0.99, 0.999]:
    print(f"{100*q:.1f}% quantile  {np.quantile(L, q):12,.0f}")
