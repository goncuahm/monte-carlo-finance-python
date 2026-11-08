import numpy as np
from mcfin_risk import var_es

# Two independent bonds, each loses 100 with probability 4% (otherwise 0)
N, p, alpha = 2_000_000, 0.04, 0.95
rng = np.random.default_rng(seed=3)
A = 100.0 * (rng.random(N) < p)
B = 100.0 * (rng.random(N) < p)
print("position      95% VaR    95% ES")
for name, L in [("bond A", A), ("bond B", B), ("A + B", A + B)]:
    v, e = var_es(L, alpha)
    print(f"{name:10s} {v:9.1f} {e:9.1f}")
vA, eA = var_es(A, alpha); vB, eB = var_es(B, alpha); vAB, eAB = var_es(A + B, alpha)
print(f"\nVaR(A) + VaR(B) = {vA + vB:.1f}  <  VaR(A+B) = {vAB:.1f}   (not subadditive)")
print(f"ES(A) + ES(B)   = {eA + eB:.1f}  >= ES(A+B)  = {eAB:.1f}   (subadditive)")
