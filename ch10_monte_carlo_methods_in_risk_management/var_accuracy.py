import numpy as np
from scipy.stats import norm
from mcfin_risk import var_es, bootstrap_ci

# How accurate is a simulated 99% VaR or ES?  Standard normal losses.
alpha = 0.99
z = norm.ppf(alpha)
true_var, true_es = z, norm.pdf(z) / (1 - alpha)
print(f"true 99% VaR {true_var:.4f}, true 99% ES {true_es:.4f}\n")
print("      N    std(VaR)  asympt.   std(ES)   tail points")
for N in [1_000, 10_000, 100_000]:
    est = np.array([var_es(np.random.default_rng(s).standard_normal(N), alpha)
                    for s in range(400)])
    asym = np.sqrt(alpha * (1 - alpha) / N) / norm.pdf(z)   # se of a sample quantile
    print(f"{N:7,d} {est[:, 0].std():9.4f} {asym:8.4f} {est[:, 1].std():9.4f} "
          f"{int(N * (1 - alpha)):8d}")
rng = np.random.default_rng(seed=99)
L = rng.standard_normal(10_000)
(v_lo, v_hi), (e_lo, e_hi) = bootstrap_ci(L, alpha, rng)
v, e = var_es(L, alpha)
print(f"\none sample of 10,000: VaR {v:.3f} [{v_lo:.3f}, {v_hi:.3f}]"
      f"   ES {e:.3f} [{e_lo:.3f}, {e_hi:.3f}]  (95% bootstrap)")
