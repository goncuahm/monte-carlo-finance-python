"""mcfin_qmc.py -- quasi-Monte Carlo tools (Chapter 7)."""
import numpy as np
from scipy.stats import norm, qmc


def van_der_corput(n, base=2):
    """First n points of the van der Corput sequence in the given base (starting at i = 1)."""
    out = np.zeros(n)
    for i in range(n):
        k, f, x = i + 1, 1.0 / base, 0.0
        while k > 0:
            k, digit = divmod(k, base)
            x += digit * f           # reflect the base-b digits about the radix point
            f /= base
        out[i] = x
    return out


def first_primes(d):
    primes, k = [], 2
    while len(primes) < d:
        if all(k % p for p in primes):
            primes.append(k)
        k += 1
    return primes


def halton(n, d):
    """First n points of the d-dimensional Halton sequence (unscrambled)."""
    return np.column_stack([van_der_corput(n, b) for b in first_primes(d)])


def sobol_normals(m, d, seed):
    """2**m scrambled Sobol points in d dimensions, mapped to standard normals."""
    U = qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)
    return norm.ppf(U)


def path_matrix(n, T, method):
    """Matrix A with W = Z @ A.T giving W(t_1..t_n) from independent normals Z.

    method: 'standard' (cumulative sums), 'bridge' (Brownian bridge, n a power of 2)
    or 'pca' (principal components of the covariance min(t_i, t_j))."""
    t = T * np.arange(1, n + 1) / n
    if method == "standard":
        return np.sqrt(T / n) * np.tril(np.ones((n, n)))
    if method == "pca":
        lam, V = np.linalg.eigh(np.minimum.outer(t, t))
        idx = np.argsort(lam)[::-1]
        return V[:, idx] * np.sqrt(lam[idx])
    # Brownian bridge: column k of A is the effect of the k-th normal on the path
    A = np.zeros((n, n))
    A[:, 0] = t / T * np.sqrt(T)              # W(T) = sqrt(T) Z_0, spread linearly
    k, step = 1, n
    while step > 1:
        half = step // 2
        for left in range(0, n, step):        # fill midpoint of (left, left + step]
            mid, right = left + half, left + step
            tl = t[left - 1] if left > 0 else 0.0
            tm, tr = t[mid - 1], t[right - 1]
            sd = np.sqrt((tm - tl) * (tr - tm) / (tr - tl))
            # the new normal moves W(mid) by sd and is interpolated linearly in between
            for j in range(left, right):
                tj = t[j]
                if tj <= tm:
                    A[j, k] = sd * (tj - tl) / (tm - tl)
                else:
                    A[j, k] = sd * (tr - tj) / (tr - tm)
            k += 1
        step = half
    return A
