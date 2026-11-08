import numpy as np
from scipy.stats import norm
from mcfin_vr import bs_call

S0, K, r, sigma, T, N = 100.0, 100.0, 0.05, 0.2, 1.0, 100_000
payoff = lambda z: np.exp(-r * T) * np.maximum(
    S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z) - K, 0)
print(f"exact {bs_call(S0, K, r, sigma, T):.4f}")

# Repeat each estimator 200 times to measure its standard deviation honestly
def repeat(estimator, n_rep=200):
    est = np.array([estimator(np.random.default_rng(1000 + i)) for i in range(n_rep)])
    return est.mean(), est.std(ddof=1)

def plain(rng):
    return payoff(rng.standard_normal(N)).mean()

def stratified(rng, M):
    # M equiprobable strata of the normal, N/M draws in each (proportional allocation)
    i = np.repeat(np.arange(M), N // M)
    U = (i + rng.random(N)) / M               # uniform in stratum i
    return payoff(norm.ppf(U)).mean()

m0, s0 = repeat(plain)
print(f"plain              mean {m0:.4f}  std.dev. {s0:.5f}")
for M in [10, 100, 1000, N]:
    m, s = repeat(lambda g: stratified(g, M))
    print(f"strata M = {M:7,d} mean {m:.4f}  std.dev. {s:.5f}  ratio {(s0 / s)**2:8.1f}")
