import numpy as np
from scipy.stats import norm, qmc

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
exact = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))
payoff = lambda z: np.exp(-r * T) * np.maximum(
    S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z) - K, 0)
print(f"Black-Scholes {exact:.6f}\n")

# Randomised QMC: R independent scramblings of a Sobol set of 2^m points
R = 16
print("    N per set   RQMC estimate   std.err.    MC std.err. (same total)  ratio")
for m in [8, 12, 16]:
    est = [payoff(norm.ppf(qmc.Sobol(1, scramble=True, seed=s).random_base2(m)[:, 0])).mean()
           for s in range(R)]
    est = np.array(est)
    se_q = est.std(ddof=1) / np.sqrt(R)
    y = payoff(np.random.default_rng(99).standard_normal(R * 2**m))
    se_mc = y.std(ddof=1) / np.sqrt(y.size)
    print(f"{2**m:10,d} {est.mean():14.6f} {se_q:11.2e} {se_mc:14.2e} {(se_mc / se_q)**2:20,.0f}")
