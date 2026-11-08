import numpy as np
from mcfin_qmc import van_der_corput

x = van_der_corput(8, base=2)
print(" i   binary   reflected   point")
for i, xi in enumerate(x, start=1):
    b = format(i, "b")
    print(f"{i:2d} {b:>8s}   0.{b[::-1]:<8s} {xi:7.4f}")
print("\nsorted first 8 points:", np.sort(x))
print("first 9 points in base 3:", np.round(van_der_corput(9, base=3), 4))
