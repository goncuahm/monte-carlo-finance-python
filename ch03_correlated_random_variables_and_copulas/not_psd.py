import numpy as np
from mcfin_mv import clip_to_correlation

# Correlations estimated pair by pair (e.g. from different time windows)
C = np.array([[ 1.0, 0.9, 0.7],
              [ 0.9, 1.0, -0.4],
              [ 0.7, -0.4, 1.0]])
print("eigenvalues:", np.linalg.eigvalsh(C).round(4))
try:
    np.linalg.cholesky(C)
except np.linalg.LinAlgError as err:
    print("Cholesky failed:", err)

C_fixed = clip_to_correlation(C)
np.set_printoptions(precision=4, suppress=True)
print("repaired matrix:\n", C_fixed)
print("eigenvalues now:", np.linalg.eigvalsh(C_fixed).round(4))
print("largest change:", np.abs(C_fixed - C).max().round(4))
