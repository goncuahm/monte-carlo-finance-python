import numpy as np
from scipy.stats import norm, t as student_t
from mcfin_risk import var_es

# Daily portfolio loss in % of value: normal and Student-t(4) with the same volatility 1%
sigma, nu, N = 1.0, 4, 1_000_000
rng = np.random.default_rng(seed=1)
L_norm = sigma * rng.standard_normal(N)
s = sigma / np.sqrt(nu / (nu - 2))                    # scale so that the std is sigma
L_t = s * rng.standard_t(nu, N)
print("alpha    model     VaR (MC)  VaR exact   ES (MC)   ES exact")
for alpha in [0.95, 0.99, 0.975]:
    z = norm.ppf(alpha)
    ex_n = (sigma * z, sigma * norm.pdf(z) / (1 - alpha))
    q = student_t.ppf(alpha, nu)
    ex_t = (s * q, s * student_t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1))
    for name, L, ex in [("normal", L_norm, ex_n), ("t(4)", L_t, ex_t)]:
        v, e = var_es(L, alpha)
        print(f"{alpha:5.3f}  {name:8s} {v:9.4f} {ex[0]:9.4f} {e:9.4f} {ex[1]:9.4f}")
