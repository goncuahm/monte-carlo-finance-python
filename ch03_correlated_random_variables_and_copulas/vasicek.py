import numpy as np
from scipy.stats import norm

# One-factor Gaussian copula: firm i defaults if sqrt(rho) M + sqrt(1 - rho) e_i < Phi^{-1}(p)
n_firms, p, lgd, n_sims = 100, 0.02, 0.6, 200_000
c = norm.ppf(p)
rng = np.random.default_rng(seed=17)

print(" rho   mean loss   std. dev.   99% quantile   99.9% quantile   (loss as % of portfolio)")
for rho in [0.0, 0.1, 0.3]:
    M = rng.standard_normal((n_sims, 1))                    # common systematic factor
    eps = rng.standard_normal((n_sims, n_firms))            # idiosyncratic factors
    defaults = (np.sqrt(rho) * M + np.sqrt(1 - rho) * eps < c).sum(axis=1)
    loss = 100 * lgd * defaults / n_firms
    q99, q999 = np.quantile(loss, [0.99, 0.999])
    print(f"{rho:4.1f}     {loss.mean():6.3f}      {loss.std():6.3f}       {q99:6.2f}          {q999:6.2f}")

# Vasicek large-portfolio formula for the default-rate quantile (basis of Basel IRB)
for rho in [0.1, 0.3]:
    x = norm.cdf((c + np.sqrt(rho) * norm.ppf(0.999)) / np.sqrt(1 - rho))
    print(f"Vasicek 99.9% loss quantile, rho = {rho}: {100 * lgd * x:.2f}%")
