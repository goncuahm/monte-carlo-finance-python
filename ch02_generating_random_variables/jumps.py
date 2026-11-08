import numpy as np

# Number of market crashes per year ~ Poisson(lam); each crash costs a lognormal fraction
rng = np.random.default_rng(seed=8)
lam, n_years = 0.5, 1_000_000
N = rng.poisson(lam, n_years)                    # jumps in each simulated year
print("jumps per year:", {k: round(float(np.mean(N == k)), 4) for k in range(4)})

# total log-jump in a year: sum of N independent normal(-0.10, 0.05^2) log-jumps
J = rng.normal(-0.10 * N, 0.05 * np.sqrt(N))     # sum of N normals, valid for N = 0 too
factor = np.exp(J)
print(f"mean price factor from jumps {factor.mean():.4f}")
print(f"theory exp(lam (e^(m + s^2/2) - 1)) = {np.exp(lam * (np.exp(-0.10 + 0.05**2/2) - 1)):.4f}")
print(f"P(jump losses > 15% in a year) {np.mean(factor < 0.85):.4f}")
