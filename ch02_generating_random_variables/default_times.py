import numpy as np

# Constant hazard rate: the default time tau is exponential(lambda).
spread, R = 0.0150, 0.40                         # 150 bp credit spread, 40% recovery
lam = spread / (1 - R)                           # 'credit triangle': lambda = s / (1 - R)
rng = np.random.default_rng(seed=31)
tau = rng.exponential(1 / lam, 1_000_000)

print(f"hazard rate lambda = {lam:.4f} per year")
for t in [1, 5, 10]:
    print(f"P(default within {t:2d} years): MC {np.mean(tau <= t):.4f}   exact {1 - np.exp(-lam * t):.4f}")

# value of a 5-year zero-coupon corporate bond paying 100, recovery R paid at default
r, T = 0.04, 5.0
pay = np.where(tau > T, 100 * np.exp(-r * T), R * 100 * np.exp(-r * tau))
print(f"corporate bond price {pay.mean():.3f}  vs risk-free {100*np.exp(-r*T):.3f}")
