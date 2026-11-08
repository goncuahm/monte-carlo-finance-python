import numpy as np
from scipy.stats import norm, binom
from scipy.integrate import quad

# One-factor Gaussian copula portfolio (Chapter 3): 100 loans, p = 2%, rho = 0.3.
# Estimate P(at least k defaults). Conditional on M the number of defaults is
# binomial, so we average P(D >= k | M) (conditional MC) and shift M (IS).
n_firms, p, rho, N = 100, 0.02, 0.3, 100_000
pM = lambda m: norm.cdf((norm.ppf(p) - np.sqrt(rho) * m) / np.sqrt(1 - rho))
exact = lambda k: quad(lambda m: binom.sf(k - 1, n_firms, pM(m)) * norm.pdf(m),
                       -12, 12, points=[-4, -3, -2, -1, 0], limit=400)[0]
print("  k   exact     plain MC (rel.err.)  cond. MC (rel.err.)  cond. + IS (rel.err.)")
for k in [10, 20, 40, 60]:
    rng = np.random.default_rng(seed=k)
    M = rng.standard_normal(N)
    D = rng.binomial(n_firms, pM(M))                   # plain: simulate the defaults
    plain = (D >= k).astype(float)
    cond = binom.sf(k - 1, n_firms, pM(M))             # exact P(D >= k | M)
    # IS: centre M where the conditional default rate equals k / n_firms
    mu = (norm.ppf(p) - np.sqrt(1 - rho) * norm.ppf(k / n_firms)) / np.sqrt(rho)
    Ms = rng.standard_normal(N) + mu
    is_est = binom.sf(k - 1, n_firms, pM(Ms)) * np.exp(-mu * Ms + 0.5 * mu**2)
    out = []
    for e in (plain, cond, is_est):
        m = e.mean()
        out.append(f"{m:9.2e} ({e.std() / np.sqrt(N) / m:6.1%})" if m > 0 else "    0     (  -  )")
    print(f"{k:3d} {exact(k):9.2e}  " + "  ".join(out))
