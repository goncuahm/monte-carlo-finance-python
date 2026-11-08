import numpy as np

rng = np.random.default_rng(seed=2)
Z = rng.standard_normal(1_000_000)

# (1) perfect dependence, zero correlation
print(f"corr(Z, Z^2) = {np.corrcoef(Z, Z**2)[0, 1]:+.4f}  although Z^2 is a function of Z")

# (2) the attainable correlation between two lognormals is limited
s1, s2 = 0.5, 1.0
for name, Z2 in [("comonotone  (Z2 =  Z)", Z), ("countermonotone (Z2 = -Z)", -Z)]:
    X1, X2 = np.exp(s1 * Z), np.exp(s2 * Z2)
    print(f"{name}: corr of lognormals {np.corrcoef(X1, X2)[0, 1]:+.3f}")
rmax = (np.exp(s1*s2) - 1) / np.sqrt((np.exp(s1**2) - 1) * (np.exp(s2**2) - 1))
rmin = (np.exp(-s1*s2) - 1) / np.sqrt((np.exp(s1**2) - 1) * (np.exp(s2**2) - 1))
print(f"theoretical range [{rmin:+.3f}, {rmax:+.3f}]")
