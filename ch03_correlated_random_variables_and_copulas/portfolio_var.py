import numpy as np
from scipy.stats import norm

V = np.array([4e6, 6e6])                          # EUR 4m and 6m in two stocks
vol = np.array([0.015, 0.020])                    # daily volatilities
rng = np.random.default_rng(seed=21)
Z = rng.standard_normal((1_000_000, 2))

print(" rho    MC 99% VaR    formula      sum of stand-alone VaRs")
standalone = (norm.ppf(0.99) * V * vol).sum()
for rho in [-0.5, 0.0, 0.5, 0.9, 1.0]:
    x2 = rho * Z[:, 0] + np.sqrt(max(1 - rho**2, 0)) * Z[:, 1]
    pnl = V[0] * vol[0] * Z[:, 0] + V[1] * vol[1] * x2
    var_mc = -np.quantile(pnl, 0.01)
    sd = np.sqrt((V[0]*vol[0])**2 + (V[1]*vol[1])**2 + 2*rho*V[0]*vol[0]*V[1]*vol[1])
    print(f"{rho:4.1f}   {var_mc:11,.0f}   {norm.ppf(0.99) * sd:11,.0f}   {standalone:11,.0f}")
