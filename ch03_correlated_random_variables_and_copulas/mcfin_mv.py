"""mcfin_mv.py -- correlated random variables and copulas (Chapter 3)."""
import numpy as np
from scipy import stats


def cholesky(A):
    """Cholesky factor L (lower triangular, A = L L^T) computed step by step."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)
    for j in range(n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s <= 0:
            raise ValueError("matrix is not positive definite")
        L[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
    return L


def correlated_normals(rng, corr, n):
    """n draws (rows) of N(0, corr) using the Cholesky factor: X = Z L^T."""
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((n, len(corr)))
    return Z @ L.T


def gaussian_copula(rng, corr, n):
    """Uniforms U = Phi(X) with X ~ N(0, corr): the Gaussian copula."""
    return stats.norm.cdf(correlated_normals(rng, corr, n))


def t_copula(rng, corr, nu, n):
    """Uniforms from the Student t copula with nu degrees of freedom."""
    X = correlated_normals(rng, corr, n)
    W = rng.chisquare(nu, size=(n, 1))          # one common chi-square per draw
    T = X / np.sqrt(W / nu)
    return stats.t.cdf(T, df=nu)


def clip_to_correlation(C, eps=1e-8):
    """Make a symmetric matrix a valid correlation matrix by clipping eigenvalues."""
    vals, vecs = np.linalg.eigh((C + C.T) / 2)
    A = vecs @ np.diag(np.maximum(vals, eps)) @ vecs.T
    d = np.sqrt(np.diag(A))
    return A / np.outer(d, d)                    # rescale to unit diagonal
