import numpy as np

# Effect of moving the barrier of a down-and-out call from 85 to 86 (daily monitoring)
S0, K, r, sigma, T, n, N = 100.0, 100.0, 0.05, 0.3, 1.0, 252, 100_000

def dao_payoffs(B, seed):
    rng = np.random.default_rng(seed)
    dt = T / n
    logS, alive = np.full(N, np.log(S0)), np.ones(N, dtype=bool)
    for _ in range(n):
        logS += (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(N)
        alive &= logS > np.log(B)
    return np.exp(-r * T) * np.maximum(np.exp(logS) - K, 0) * alive

for label, s1, s2 in [("independent streams", 11, 12), ("common random numbers", 11, 11)]:
    D = dao_payoffs(86, s2) - dao_payoffs(85, s1)
    print(f"{label:22s} price change {D.mean():+.4f}   s.e. {D.std()/np.sqrt(N):.4f}")
