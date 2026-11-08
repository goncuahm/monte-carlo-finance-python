import numpy as np

# Two stocks with correlated Brownian motions; spread option on S1 - S2 at T
S0, r, T, n_steps, n_paths = np.array([100.0, 95.0]), 0.03, 1.0, 52, 200_000
sig, rho = np.array([0.25, 0.20]), 0.5
L = np.linalg.cholesky([[1, rho], [rho, 1]])
dt = T / n_steps
rng = np.random.default_rng(seed=50)
logS = np.tile(np.log(S0), (n_paths, 1))
rets = []
for _ in range(n_steps):
    dZ = rng.standard_normal((n_paths, 2)) @ L.T          # correlated increments
    step = (r - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * dZ
    logS += step
    rets.append(step[:1000])
S = np.exp(logS)
R = np.concatenate(rets)
print(f"correlation of weekly log-returns {np.corrcoef(R[:, 0], R[:, 1])[0, 1]:.4f}")
for K in [0, 5, 10]:
    pay = np.exp(-r * T) * np.maximum(S[:, 0] - S[:, 1] - K, 0)
    print(f"spread call K = {K:2d}: {pay.mean():.4f} +/- {1.96*pay.std()/np.sqrt(n_paths):.4f}")
