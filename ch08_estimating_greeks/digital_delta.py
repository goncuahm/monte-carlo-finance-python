import numpy as np
from mcfin_greeks import digital_greeks, terminal, mean_se

S0, K, r, sigma, T, N = 100.0, 100.0, 0.05, 0.2, 1.0, 100_000
exact = digital_greeks(S0, K, r, sigma, T)["delta"]
rng = np.random.default_rng(seed=6)
Z = rng.standard_normal(N)
disc = np.exp(-r * T)
dig = lambda s0: disc * (terminal(s0, r, sigma, T, Z) > K)
print(f"exact digital delta {exact:.5f}\n")
for h in [2.0, 0.5, 0.1, 0.01]:
    m, s = mean_se((dig(S0 + h) - dig(S0 - h)) / (2 * h))
    nz = np.mean(dig(S0 + h) != dig(S0 - h))
    print(f"FD/CRN h = {h:5.2f}   {m:.5f}  s.e. {s:.5f}   paths that change: {nz:.4f}")
print(f"pathwise             {0.0:.5f}  (derivative of an indicator is zero a.e.)")
m, s = mean_se(dig(S0) * Z / (S0 * sigma * np.sqrt(T)))
print(f"likelihood ratio     {m:.5f}  s.e. {s:.5f}")
