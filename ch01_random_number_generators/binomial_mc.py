import numpy as np
from math import comb, exp

S0, K, r, T = 100.0, 100.0, 0.05, 1.0        # spot, strike, rate, maturity (years)
n, u_fac, d_fac = 12, 1.06, 1 / 1.06          # 12 monthly steps
dt = T / n
q = (exp(r * dt) - d_fac) / (u_fac - d_fac)   # risk-neutral up probability

# exact price from the binomial formula
exact = exp(-r * T) * sum(comb(n, j) * q**j * (1 - q)**(n - j)
                          * max(S0 * u_fac**j * d_fac**(n - j) - K, 0) for j in range(n + 1))

# Monte Carlo: each step goes up when a uniform number is below q
rng = np.random.default_rng(seed=8)
N = 200_000
ups = (rng.random((N, n)) < q).sum(axis=1)
ST = S0 * u_fac**ups * d_fac**(n - ups)
payoff = np.maximum(ST - K, 0)
mc = exp(-r * T) * payoff.mean()
se = exp(-r * T) * payoff.std(ddof=1) / np.sqrt(N)
print(f"risk-neutral q = {q:.4f}")
print(f"exact binomial price  {exact:.4f}")
print(f"Monte Carlo price     {mc:.4f}  +/- {1.96 * se:.4f} (95% interval)")
