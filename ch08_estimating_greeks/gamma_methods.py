import numpy as np
from mcfin_greeks import bs_greeks, terminal, mean_se

S0, K, r, sigma, T, N = 100.0, 100.0, 0.05, 0.2, 1.0, 100_000
exact = bs_greeks(S0, K, r, sigma, T)["gamma"]
rng = np.random.default_rng(seed=5)
Z = rng.standard_normal(N)
disc, sT = np.exp(-r * T), sigma * np.sqrt(T)
call = lambda s0: disc * np.maximum(terminal(s0, r, sigma, T, Z) - K, 0)
print(f"exact gamma {exact:.5f}\n")
for h in [5.0, 1.0, 0.1]:
    m, s = mean_se((call(S0 + h) - 2 * call(S0) + call(S0 - h)) / h**2)
    print(f"FD with CRN, h = {h:4.1f}     {m:.5f}  s.e. {s:.5f}")
ST = terminal(S0, r, sigma, T, Z)
m, s = mean_se(disc * np.maximum(ST - K, 0) * ((Z**2 - 1) / sT - Z) / (S0**2 * sT))
print(f"likelihood ratio          {m:.5f}  s.e. {s:.5f}")
phi = disc * (ST > K) * ST                       # pathwise delta times S0
m, s = mean_se(phi * (Z / sT - 1) / S0**2)       # LR applied to the pathwise delta
print(f"mixed pathwise-LR         {m:.5f}  s.e. {s:.5f}")
