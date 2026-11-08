import numpy as np
from scipy import stats
from mcfin_mv import gaussian_copula

# Two daily returns with Student t(4) marginals (vol 1.5% and 2%) and a Gaussian copula
corr = np.array([[1.0, 0.6], [0.6, 1.0]])
U = gaussian_copula(np.random.default_rng(seed=13), corr, 500_000)
nu = 4
scale = np.sqrt((nu - 2) / nu)
R1 = 0.015 * scale * stats.t.ppf(U[:, 0], nu)       # inverse transform of each marginal
R2 = 0.020 * scale * stats.t.ppf(U[:, 1], nu)

print(f"vols {R1.std():.4f}, {R2.std():.4f}   excess kurtosis {stats.kurtosis(R1):.1f}, {stats.kurtosis(R2):.1f}")
print(f"Spearman correlation {stats.spearmanr(R1[:50_000], R2[:50_000]).statistic:.3f}"
      f"  (copula value {6/np.pi*np.arcsin(0.3):.3f})")
print(f"KS test of R1 against its t marginal: p-value "
      f"{stats.kstest(R1 / (0.015 * scale), 't', args=(nu,)).pvalue:.3f}")
port = 0.5 * R1 + 0.5 * R2
print(f"99% VaR of an equally weighted portfolio: {-np.quantile(port, 0.01):.4f}")
