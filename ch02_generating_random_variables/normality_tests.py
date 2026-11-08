import numpy as np
from scipy import stats

rng = np.random.default_rng(seed=9)
samples = {"normal": rng.standard_normal(2_000),
           "Student t(4)": rng.standard_t(4, 2_000) * np.sqrt(2 / 4)}
for name, x in samples.items():
    jb = stats.jarque_bera(x)
    print(f"{name:13s} skew {stats.skew(x):+.3f}  excess kurtosis {stats.kurtosis(x):+.3f}  "
          f"JB = {jb.statistic:7.1f}  p-value = {jb.pvalue:.3f}")
