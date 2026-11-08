import time
import numpy as np
from scipy import stats
from mcfin_rv import box_muller, marsaglia_polar, norm_ppf

rng = np.random.default_rng(seed=2026)
n = 1_000_000
methods = {"Box-Muller": lambda: box_muller(rng, n),
           "Marsaglia polar": lambda: marsaglia_polar(rng, n),
           "inverse CDF (Acklam)": lambda: norm_ppf(rng.random(n)),
           "numpy standard_normal": lambda: rng.standard_normal(n)}
print("method                  mean     var    skew   ex.kurt    time (s)")
for name, gen in methods.items():
    t0 = time.perf_counter(); z = gen(); dt = time.perf_counter() - t0
    print(f"{name:22s} {z.mean():+.4f}  {z.var():.4f}  {stats.skew(z):+.4f}  {stats.kurtosis(z):+.4f}   {dt:.3f}")

# accuracy of Acklam's approximation against SciPy's inverse normal CDF
p = np.linspace(1e-10, 1 - 1e-10, 2_000_001)
err = np.abs(norm_ppf(p) - stats.norm.ppf(p)) / np.abs(stats.norm.ppf(p) + 1e-300)
print(f"max relative error of norm_ppf: {err[np.abs(stats.norm.ppf(p)) > 1e-3].max():.2e}")
