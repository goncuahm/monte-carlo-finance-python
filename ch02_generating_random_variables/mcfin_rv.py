"""mcfin_rv.py -- methods for generating random variables (Chapter 2).

Every function takes a NumPy Generator `rng` so that results are reproducible.
"""
import numpy as np


def exponential_inv(rng, lam, size):
    """Exponential(lam) by inversion: X = -ln(1 - U) / lam."""
    u = rng.random(size)
    return -np.log1p(-u) / lam


def discrete_inv(rng, values, probs, size):
    """Inverse transform for a discrete distribution P(X = values[j]) = probs[j]."""
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0                                   # guard against rounding
    u = rng.random(size)
    return np.asarray(values)[np.searchsorted(cdf, u, side="right")]


def accept_reject(rng, f, g_sample, g_pdf, c, size):
    """Acceptance-rejection: sample from density f using proposals from g, f <= c g."""
    out = np.empty(0)
    tries = 0
    while out.size < size:
        n = int(1.2 * c * (size - out.size)) + 10   # proposals for this round
        y = g_sample(rng, n)
        u = rng.random(n)
        out = np.concatenate([out, y[u <= f(y) / (c * g_pdf(y))]])
        tries += n
    return out[:size], out.size / tries             # sample and acceptance rate


def box_muller(rng, size):
    """Standard normals by the Box-Muller transform (size must be even)."""
    u1, u2 = rng.random(size // 2), rng.random(size // 2)
    r = np.sqrt(-2.0 * np.log1p(-u1))               # 1 - u1 is in (0, 1]
    return np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])


def marsaglia_polar(rng, size):
    """Standard normals by Marsaglia's polar method (no sine or cosine)."""
    out = []
    count = 0
    while count < size:
        v = 2.0 * rng.random((size, 2)) - 1.0
        s = (v ** 2).sum(axis=1)
        ok = (s > 0) & (s < 1)                      # keep points inside the unit disc
        v, s = v[ok], s[ok]
        factor = np.sqrt(-2.0 * np.log(s) / s)
        z = np.concatenate([v[:, 0] * factor, v[:, 1] * factor])
        out.append(z)
        count += z.size
    return np.concatenate(out)[:size]


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's algorithm, relative error < 1.2e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p = np.asarray(p, dtype=float)
    x = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))                 # lower tail
    x[lo] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p[mid] - 0.5                                # central region
    r = q * q
    x[mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
             (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))             # upper tail, by symmetry
    x[hi] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    return x
