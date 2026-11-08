import numpy as np

# Four assets: annual volatilities and correlations
vol = np.array([0.20, 0.25, 0.15, 0.30])
corr = np.array([[1.0, 0.6, 0.3, 0.2],
                 [0.6, 1.0, 0.4, 0.3],
                 [0.3, 0.4, 1.0, 0.1],
                 [0.2, 0.3, 0.1, 1.0]])
cov = np.outer(vol, vol) * corr                   # Sigma_ij = sigma_i sigma_j rho_ij
mu = np.array([0.06, 0.08, 0.04, 0.10])

rng = np.random.default_rng(seed=7)
L = np.linalg.cholesky(cov)
R = mu + rng.standard_normal((1_000_000, 4)) @ L.T  # annual returns, one row per scenario

np.set_printoptions(precision=4, suppress=True)
print("sample means:   ", R.mean(axis=0))
print("sample vols:    ", R.std(axis=0))
print("sample correlation matrix:\n", np.corrcoef(R, rowvar=False))
print("max abs error of the covariance:", np.abs(np.cov(R, rowvar=False) - cov).max().round(5))
