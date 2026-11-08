import numpy as np
from mcfin_greeks import bs_greeks, terminal, mean_se

S0, K, r, sigma, T, N = 100.0, 100.0, 0.05, 0.2, 1.0, 100_000
exact = bs_greeks(S0, K, r, sigma, T)["delta"]
call = lambda s0, Z: np.exp(-r * T) * np.maximum(terminal(s0, r, sigma, T, Z) - K, 0)
print(f"exact delta {exact:.5f}\n")
print("     h    forward/indep.    forward/CRN       central/CRN")
for h in [5.0, 1.0, 0.1, 0.01]:
    rng = np.random.default_rng(seed=1)
    Z1, Z2 = rng.standard_normal(N), rng.standard_normal(N)
    fwd_ind = (call(S0 + h, Z1) - call(S0, Z2)) / h          # independent draws
    fwd_crn = (call(S0 + h, Z1) - call(S0, Z1)) / h          # same draws
    cen_crn = (call(S0 + h, Z1) - call(S0 - h, Z1)) / (2 * h)
    out = [mean_se(x) for x in (fwd_ind, fwd_crn, cen_crn)]
    print(f"{h:6.2f} " + "  ".join(f"{m:8.4f} ({s:.4f})" for m, s in out))
