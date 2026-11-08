import numpy as np
from scipy import stats

rng = np.random.default_rng(seed=17)
n, nu = 400_000, 5
Z = rng.standard_normal((n, nu))
chi2 = (Z ** 2).sum(axis=1)                       # chi-square with nu degrees of freedom
t = rng.standard_normal(n) / np.sqrt(chi2 / nu)   # Student t with nu degrees of freedom

print(f"chi2({nu}): mean {chi2.mean():.3f} (theory {nu}), var {chi2.var():.3f} (theory {2*nu})")
print(f"t({nu}):    var {t.var():.3f} (theory nu/(nu-2) = {nu/(nu-2):.3f}), "
      f"excess kurtosis {stats.kurtosis(t):.2f} (theory 6/(nu-4) = {6/(nu-4):.0f})")
for q in [0.95, 0.99, 0.999]:
    print(f"  {q:.3f} quantile:  t({nu}) MC {np.quantile(t, q):.3f}  exact {stats.t.ppf(q, nu):.3f}"
          f"   normal {stats.norm.ppf(q):.3f}")
