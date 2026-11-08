import numpy as np
from scipy import stats

# Five firms, hazard rate 2% each, default times linked by a Gaussian copula
k, lam, T, n = 5, 0.02, 5.0, 400_000
print(" rho   P(at least 1 default)   P(all 5 default)   E[number of defaults]")
for rho in [0.0, 0.3, 0.6, 0.9]:
    corr = np.full((k, k), rho) + (1 - rho) * np.eye(k)
    X = np.random.default_rng(3).standard_normal((n, k)) @ np.linalg.cholesky(corr).T
    U = stats.norm.cdf(X)
    tau = -np.log1p(-U) / lam                     # exponential marginals by inversion
    d = (tau <= T).sum(axis=1)
    print(f"{rho:4.1f}        {np.mean(d >= 1):.4f}               {np.mean(d == k):.5f}            {d.mean():.4f}")
print(f"single-name 5-year default probability {1 - np.exp(-lam * T):.4f}")
