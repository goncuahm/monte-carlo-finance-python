import numpy as np
from mcfin_mv import cholesky

C = np.array([[1.0, 0.6, 0.3],
              [0.6, 1.0, 0.5],
              [0.3, 0.5, 1.0]])
L = cholesky(C)
np.set_printoptions(precision=4, suppress=True)
print("our Cholesky factor L:\n", L)
print("numpy.linalg.cholesky agrees:", np.allclose(L, np.linalg.cholesky(C)))
print("L @ L.T reproduces C:        ", np.allclose(L @ L.T, C))
