import numpy as np
from mcfin_vr import bs_call

# Deep out-of-the-money call: shift the normal so that S(T) = K on average
S0, r, sigma, T, N = 100.0, 0.05, 0.2, 1.0, 100_000
print("   K    exact      plain (s.e.)      IS (s.e.)      P(ITM)   ratio")
for K in [120, 140, 160, 180]:
    rng = np.random.default_rng(seed=K)
    S = lambda z: S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    Y = np.exp(-r * T) * np.maximum(S(rng.standard_normal(N)) - K, 0)
    mu = (np.log(K / S0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    Zs = rng.standard_normal(N) + mu
    Yis = np.exp(-r * T) * np.maximum(S(Zs) - K, 0) * np.exp(-mu * Zs + 0.5 * mu**2)
    s1, s2 = Y.std() / np.sqrt(N), Yis.std() / np.sqrt(N)
    print(f"{K:5d} {bs_call(S0, K, r, sigma, T):9.5f} {Y.mean():9.5f} ({s1:.5f})"
          f" {Yis.mean():9.5f} ({s2:.5f}) {np.mean(Y > 0):8.4f} {(s1 / s2)**2:7.1f}")
