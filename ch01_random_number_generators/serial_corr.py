import numpy as np
from mcfin_rng import MINSTD, lcg_uniform

def lag_corr(u, lag=1):
    return np.corrcoef(u[:-lag], u[lag:])[0, 1]

good = np.array(MINSTD(seed=5).sample(50_000))
bad = np.array(lcg_uniform(seed=1, a=65, c=1, m=2**16, n=50_000))
for lag in [1, 2, 3]:
    print(f"lag {lag}:  MINSTD {lag_corr(good, lag):+.4f}    a=65, m=2^16 {lag_corr(bad, lag):+.4f}")
print(f"approximate 95% band for independent data: +/-{1.96/np.sqrt(good.size):.4f}")
