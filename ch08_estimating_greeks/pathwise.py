import numpy as np
from mcfin_greeks import bs_greeks, terminal, mean_se

S0, K, r, sigma, T, N = 100.0, 100.0, 0.05, 0.2, 1.0, 100_000
g = bs_greeks(S0, K, r, sigma, T)
rng = np.random.default_rng(seed=3)
Z = rng.standard_normal(N)
ST = terminal(S0, r, sigma, T, Z)
disc, itm = np.exp(-r * T), (ST > K)
delta_pw = disc * itm * ST / S0                                   # dS_T/dS0 = S_T/S0
vega_pw = disc * itm * ST * (np.sqrt(T) * Z - sigma * T)          # dS_T/dsigma
for name, Y, ex in [("delta", delta_pw, g["delta"]), ("vega", vega_pw, g["vega"])]:
    m, s = mean_se(Y)
    print(f"pathwise {name:5s} {m:9.4f}  s.e. {s:.4f}   exact {ex:9.4f}")
# Same draws, central finite differences with h = 0.01 for comparison
call = lambda s0, sig: disc * np.maximum(terminal(s0, r, sig, T, Z) - K, 0)
print(f"FD/CRN   delta {np.mean((call(S0+.01, sigma) - call(S0-.01, sigma)) / .02):9.4f}")
print(f"FD/CRN   vega  {np.mean((call(S0, sigma+1e-4) - call(S0, sigma-1e-4)) / 2e-4):9.4f}")
