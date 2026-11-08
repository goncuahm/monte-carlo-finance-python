import numpy as np
from mcfin_greeks import mean_se
from scipy.stats import norm

# Delta of a down-and-out call, 52 weekly monitoring dates
S0, K, B, r, sigma, T, n, N = 100.0, 100.0, 85.0, 0.05, 0.3, 1.0, 52, 400_000
h = T / n
rng = np.random.default_rng(seed=8)
Z = rng.standard_normal((N, n))
inc = (r - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * Z
def payoff(s0):
    logS = np.log(s0) + np.cumsum(inc, axis=1)
    alive = (logS > np.log(B)).all(axis=1)
    return np.exp(-r * T) * np.maximum(np.exp(logS[:, -1]) - K, 0) * alive
pay = payoff(S0)
print(f"price {pay.mean():.4f}\n")
for dS in [2.0, 0.5, 0.1]:
    m, s = mean_se((payoff(S0 + dS) - payoff(S0 - dS)) / (2 * dS))
    print(f"FD/CRN  h = {dS:4.1f}   delta {m:.4f}  s.e. {s:.4f}")
m, s = mean_se(pay * Z[:, 0] / (S0 * sigma * np.sqrt(h)))
print(f"likelihood ratio     delta {m:.4f}  s.e. {s:.4f}")
# Pathwise derivative ignores the knock-out discontinuity -> biased
itm = pay > 0
ST = S0 * np.exp(inc.sum(axis=1))
m, s = mean_se(np.exp(-r * T) * itm * ST / S0)
print(f"pathwise (biased)    delta {m:.4f}  s.e. {s:.4f}")
