import numpy as np
from scipy.stats import norm

# Exchange option: the right to receive asset 1 and give asset 2, payoff max(S1 - S2, 0)
S1, S2, s1, s2, rho, r, T = 100.0, 95.0, 0.25, 0.20, 0.5, 0.04, 1.0
s = np.sqrt(s1**2 + s2**2 - 2 * rho * s1 * s2)
d1 = (np.log(S1 / S2) + 0.5 * s**2 * T) / (s * np.sqrt(T))
exact = S1 * norm.cdf(d1) - S2 * norm.cdf(d1 - s * np.sqrt(T))    # Margrabe (1978)

rng = np.random.default_rng(seed=5)
n = 1_000_000
Z1 = rng.standard_normal(n)
Z2 = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
X1 = S1 * np.exp((r - 0.5 * s1**2) * T + s1 * np.sqrt(T) * Z1)
X2 = S2 * np.exp((r - 0.5 * s2**2) * T + s2 * np.sqrt(T) * Z2)
pay = np.exp(-r * T) * np.maximum(X1 - X2, 0)
mc, se = pay.mean(), pay.std(ddof=1) / np.sqrt(n)
print(f"Margrabe formula  {exact:.4f}")
print(f"Monte Carlo       {mc:.4f}  +/- {1.96 * se:.4f}")
