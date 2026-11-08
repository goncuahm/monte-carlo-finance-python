import numpy as np
from scipy import stats
from mcfin_rv import exponential_inv

rng = np.random.default_rng(seed=1)
lam = 2.0                                        # rate; mean 1/lam = 0.5
x = exponential_inv(rng, lam, 100_000)
print(f"sample mean     {x.mean():.4f}   (theory 1/lam   = {1/lam:.4f})")
print(f"sample variance {x.var(ddof=1):.4f}   (theory 1/lam^2 = {1/lam**2:.4f})")
print(f"P(X > 1)        {np.mean(x > 1):.4f}   (theory e^-2    = {np.exp(-2):.4f})")
ks = stats.kstest(x, "expon", args=(0, 1 / lam))
print(f"KS test: D = {ks.statistic:.4f}, p-value = {ks.pvalue:.3f}")
