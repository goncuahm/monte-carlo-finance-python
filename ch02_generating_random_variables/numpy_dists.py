import numpy as np

rng = np.random.default_rng(seed=1)
print("normal(0.05, 0.2):     ", rng.normal(0.05, 0.2, 3).round(4))
print("exponential(scale=2):  ", rng.exponential(2.0, 3).round(4))     # scale = 1/lambda
print("poisson(4):            ", rng.poisson(4.0, 6))
print("binomial(10, 0.3):     ", rng.binomial(10, 0.3, 6))
print("standard_t(4):         ", rng.standard_t(4, 3).round(4))
print("chisquare(5):          ", rng.chisquare(5, 3).round(4))
# lognormal(mean, sigma): mean and sigma of log X, NOT of X itself
x = rng.lognormal(mean=0.0, sigma=0.5, size=1_000_000)
print(f"lognormal(0, 0.5): E[X] = {x.mean():.4f} (theory e^0.125 = {np.exp(0.125):.4f})")
