import numpy as np
from scipy.stats import norm

# P(Z > c) for Z ~ N(0,1): plain Monte Carlo versus sampling from N(mu, 1)
N = 1_000_000
print("  c    exact         plain (rel.err.)       IS, mu = c (rel.err.)")
for c in [2, 3, 4, 5, 6]:
    rng = np.random.default_rng(seed=c)
    Z = rng.standard_normal(N)
    hit = (Z > c).astype(float)
    p_plain = hit.mean()
    re_plain = hit.std() / np.sqrt(N) / p_plain if p_plain > 0 else np.inf
    mu = c
    Y = rng.standard_normal(N) + mu                    # draws from N(mu, 1)
    w = np.exp(-mu * Y + 0.5 * mu**2)                  # likelihood ratio phi(y)/phi(y-mu)
    est = (Y > c) * w
    p_is = est.mean()
    print(f"{c:3d} {norm.sf(c):10.3e} {p_plain:10.3e} ({re_plain:7.2%}) "
          f"{p_is:12.3e} ({est.std() / np.sqrt(N) / p_is:6.2%})")
