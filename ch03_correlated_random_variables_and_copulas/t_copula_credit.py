import numpy as np
from scipy.stats import norm, t as student_t

# Same default probability and correlation; Gaussian versus t copula (nu = 4)
n_firms, p, lgd, rho, nu, n_sims = 100, 0.02, 0.6, 0.2, 4, 200_000
rng = np.random.default_rng(seed=23)
M = rng.standard_normal((n_sims, 1))
eps = rng.standard_normal((n_sims, n_firms))
X = np.sqrt(rho) * M + np.sqrt(1 - rho) * eps

gauss = (X < norm.ppf(p)).sum(axis=1)
W = rng.chisquare(nu, size=(n_sims, 1))
T = X / np.sqrt(W / nu)                                     # common mixing variable
tcop = (T < student_t.ppf(p, nu)).sum(axis=1)

for name, d in [("Gaussian copula", gauss), ("t copula (nu=4)", tcop)]:
    loss = 100 * lgd * d / n_firms
    print(f"{name}:  mean {loss.mean():.3f}%   99% {np.quantile(loss, 0.99):5.2f}%   "
          f"99.9% {np.quantile(loss, 0.999):5.2f}%   P(no default) {np.mean(d == 0):.3f}")
