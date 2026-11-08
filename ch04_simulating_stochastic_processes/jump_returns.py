import numpy as np
from scipy import stats
from mcfin_sp import merton_paths

# Daily log-returns over 50 years under GBM (sigma 25%) and Merton jump-diffusion
rng = np.random.default_rng(seed=14)
n = 252 * 50
_, S = merton_paths(rng, 100, 0.03, 0.15, 2.0, -0.10, 0.08, T=50, n_steps=n, n_paths=1)
ret_j = np.diff(np.log(S[0]))
ret_n = 0.2352 / np.sqrt(252) * rng.standard_normal(n)       # same total volatility
print("                 GBM     Merton")
print(f"daily std     {ret_n.std():7.4f}  {ret_j.std():8.4f}")
print(f"skewness      {stats.skew(ret_n):7.2f}  {stats.skew(ret_j):8.2f}")
print(f"excess kurt.  {stats.kurtosis(ret_n):7.2f}  {stats.kurtosis(ret_j):8.2f}")
for k in [4, 6]:
    print(f"days < -{k} sd  {np.sum(ret_n < -k*ret_n.std()):7d}  {np.sum(ret_j < -k*ret_j.std()):8d}")
