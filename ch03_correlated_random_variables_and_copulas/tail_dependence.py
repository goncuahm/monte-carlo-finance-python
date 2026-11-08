import numpy as np
from scipy import stats
from mcfin_mv import gaussian_copula, t_copula

rho, nu, n = 0.7, 3, 4_000_000
corr = np.array([[1.0, rho], [rho, 1.0]])
G = gaussian_copula(np.random.default_rng(1), corr, n)
T = t_copula(np.random.default_rng(1), corr, nu, n)

print("    q     P(U2 > q | U1 > q)   Gaussian      t(3)")
for q in [0.90, 0.95, 0.99, 0.999]:
    g = np.mean((G[:, 0] > q) & (G[:, 1] > q)) / (1 - q)
    t = np.mean((T[:, 0] > q) & (T[:, 1] > q)) / (1 - q)
    print(f"{q:6.3f}                        {g:8.3f}    {t:8.3f}")
lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)
print(f"limit q -> 1:                    0.000    {lam:8.3f}   (tail dependence coefficients)")
