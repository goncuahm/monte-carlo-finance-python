import numpy as np
from mcfin_vr import bs_call, summary, control_variate

S0, r, sigma, T, N = 100.0, 0.05, 0.2, 1.0, 100_000
rng = np.random.default_rng(seed=3)
ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * rng.standard_normal(N))
X = np.exp(-r * T) * ST                     # control: discounted stock, E[X] = S0
print("  K   exact    plain (s.e.)       control variate (s.e.)    b    corr   ratio")
for K in [60, 80, 100, 120, 140]:
    Y = np.exp(-r * T) * np.maximum(ST - K, 0)
    Y_cv, b = control_variate(Y, X, S0)
    (p, s, _), (pc, sc, _) = summary(Y), summary(Y_cv)
    print(f"{K:4d} {bs_call(S0, K, r, sigma, T):7.4f} {p:8.4f} ({s:.4f}) {pc:12.4f} ({sc:.4f})"
          f" {b[0]:8.3f} {np.corrcoef(X, Y)[0, 1]:6.3f} {(s / sc)**2:6.1f}")
