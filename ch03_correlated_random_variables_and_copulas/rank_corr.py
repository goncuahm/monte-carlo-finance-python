import numpy as np
from scipy import stats

rng = np.random.default_rng(seed=6)
n = 20_000
print(" rho    Pearson   Spearman (theory)    Kendall (theory)")
for rho in [0.3, 0.6, 0.9]:
    Z1 = rng.standard_normal(n)
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    X1, X2 = np.exp(Z1), np.exp(2 * Z2)           # monotone transformations
    sp = stats.spearmanr(X1, X2).statistic
    kt = stats.kendalltau(X1, X2).statistic
    print(f"{rho:4.1f}   {np.corrcoef(X1, X2)[0,1]:.3f}     {sp:.3f} ({6/np.pi*np.arcsin(rho/2):.3f})"
          f"      {kt:.3f} ({2/np.pi*np.arcsin(rho):.3f})")
