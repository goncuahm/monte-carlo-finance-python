import numpy as np
from mcfin_sp import merton_paths, merton_call, bs_call, implied_vol

S0, r, sigma, lam, muJ, delJ, T = 100.0, 0.03, 0.15, 2.0, -0.10, 0.08, 1.0
rng = np.random.default_rng(seed=13)
# Only S(T) is needed, so one "step" suffices: the scheme is exact in distribution
_, S = merton_paths(rng, S0, r, sigma, lam, muJ, delJ, T, n_steps=1, n_paths=2_000_000)
ST = S[:, -1]
print(f"E[S(T)] e^(-rT) = {np.exp(-r*T)*ST.mean():.4f}   (martingale check, S0 = 100)")
tot_vol = np.sqrt(sigma**2 + lam * (muJ**2 + delJ**2))
print(f"total volatility of log-returns {np.log(ST).std():.4f}  (theory {tot_vol:.4f})")
print("\nstrike   MC price  std.err.  Merton    BS(total vol)  implied vol")
for K in [80, 90, 100, 110, 120]:
    pay = np.exp(-r * T) * np.maximum(ST - K, 0)
    mp = merton_call(S0, K, r, sigma, lam, muJ, delJ, T)
    print(f"{K:5d} {pay.mean():10.4f} {pay.std()/np.sqrt(ST.size):8.4f} {mp:9.4f}"
          f" {bs_call(S0, K, r, tot_vol, T):11.4f} {implied_vol(mp, S0, K, r, T):12.4f}")
