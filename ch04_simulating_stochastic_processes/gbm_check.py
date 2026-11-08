import numpy as np
from mcfin_sp import gbm_paths

S0, mu, sigma, T = 100.0, 0.08, 0.25, 2.0
rng = np.random.default_rng(seed=12)
t, S = gbm_paths(rng, S0, mu, sigma, T, n_steps=500, n_paths=20_000)
ST = S[:, -1]
logret = np.diff(np.log(S), axis=1)
print(f"mean S(T)      {ST.mean():8.3f}   theory {S0*np.exp(mu*T):8.3f}")
print(f"median S(T)    {np.median(ST):8.3f}   theory {S0*np.exp((mu-sigma**2/2)*T):8.3f}")
print(f"std of log S(T) {np.log(ST).std():7.4f}   theory {sigma*np.sqrt(T):8.4f}")
print(f"P(S(T) < S0)   {np.mean(ST < S0):8.4f}")
print(f"annualised vol of step log-returns {logret.std() * np.sqrt(500 / T):.4f}")
