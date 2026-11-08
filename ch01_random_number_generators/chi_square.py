import numpy as np
from scipy import stats
from mcfin_rng import MINSTD

u = np.array(MINSTD(seed=99).sample(10_000))
k = 10                                            # number of equal bins
observed, _ = np.histogram(u, bins=k, range=(0, 1))
expected = np.full(k, u.size / k)
chi2 = ((observed - expected) ** 2 / expected).sum()
p_value = stats.chi2.sf(chi2, df=k - 1)
print("observed counts:", observed)
print(f"chi-square = {chi2:.3f}, degrees of freedom = {k-1}, p-value = {p_value:.3f}")

ks = stats.kstest(u, "uniform")
print(f"Kolmogorov-Smirnov D = {ks.statistic:.4f}, p-value = {ks.pvalue:.3f}")

# a deliberately bad 'generator': squares of uniforms are not uniform
bad = u ** 2
obs_bad, _ = np.histogram(bad, bins=k, range=(0, 1))
chi2_bad = ((obs_bad - expected) ** 2 / expected).sum()
print(f"u^2: chi-square = {chi2_bad:.1f}, p-value = {stats.chi2.sf(chi2_bad, k-1):.2e}")
