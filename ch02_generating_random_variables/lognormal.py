import numpy as np

rng = np.random.default_rng(seed=3)
S0, r, sigma, T = 100.0, 0.05, 0.20, 1.0
n = 1_000_000
Z = rng.standard_normal(n)
ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)   # risk-neutral GBM

print(f"E[S_T]      MC {ST.mean():8.4f}   theory S0 e^(rT) = {S0*np.exp(r*T):.4f}")
print(f"sd(S_T)     MC {ST.std():8.4f}   theory {S0*np.exp(r*T)*np.sqrt(np.exp(sigma**2*T)-1):.4f}")
print(f"median      MC {np.median(ST):8.4f}   theory {S0*np.exp((r-0.5*sigma**2)*T):.4f}")
print(f"P(S_T < 80) MC {np.mean(ST < 80):8.4f}")
print(f"min, max    {ST.min():.2f}, {ST.max():.2f}   (always positive)")
