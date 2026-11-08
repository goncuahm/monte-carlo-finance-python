import numpy as np
from mcfin_greeks import bs_greeks, terminal, mean_se

S0, K, r, sigma, T, N = 100.0, 100.0, 0.05, 0.2, 1.0, 100_000
g = bs_greeks(S0, K, r, sigma, T)
rng = np.random.default_rng(seed=4)
Z = rng.standard_normal(N)
ST = terminal(S0, r, sigma, T, Z)
pay = np.exp(-r * T) * np.maximum(ST - K, 0)
sT = sigma * np.sqrt(T)
scores = {"delta": Z / (S0 * sT),
          "vega": (Z**2 - 1) / sigma - Z * np.sqrt(T),
          "gamma": ((Z**2 - 1) / sT - Z) / (S0**2 * sT)}
pw = {"delta": np.exp(-r * T) * (ST > K) * ST / S0,
      "vega": np.exp(-r * T) * (ST > K) * ST * (np.sqrt(T) * Z - sigma * T)}
print("greek     exact     LR (s.e.)           pathwise (s.e.)")
for k, score in scores.items():
    m, s = mean_se(pay * score)
    line = f"{k:6s} {g[k]:9.4f} {m:9.4f} ({s:.4f})"
    if k in pw:
        mp, sp = mean_se(pw[k]); line += f"   {mp:9.4f} ({sp:.4f})"
    print(line)
