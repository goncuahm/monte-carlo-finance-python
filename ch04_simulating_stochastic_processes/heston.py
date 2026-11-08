import numpy as np
from scipy.integrate import quad
from mcfin_sp import bs_call, implied_vol

S0, r, T = 100.0, 0.03, 1.0
v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.6, -0.7    # Heston parameters

def heston_cf(u):
    """Characteristic function E[exp(i u ln S_T)] of the Heston model."""
    b = kappa - 1j * rho * xi * u
    d = np.sqrt(b**2 + xi**2 * (1j * u + u**2))
    g = (b - d) / (b + d)
    C = 1j * u * (np.log(S0) + r * T) + kappa * theta / xi**2 * (
        (b - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = (b - d) / xi**2 * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    return np.exp(C + D * v0)

def heston_call(K):
    k = np.log(K)
    f1 = lambda u: (np.exp(-1j*u*k) * heston_cf(u - 1j) / (1j*u*S0*np.exp(r*T))).real
    f2 = lambda u: (np.exp(-1j*u*k) * heston_cf(u) / (1j*u)).real
    P1 = 0.5 + quad(f1, 1e-8, 200, limit=200)[0] / np.pi
    P2 = 0.5 + quad(f2, 1e-8, 200, limit=200)[0] / np.pi
    return S0 * P1 - K * np.exp(-r * T) * P2

# Monte Carlo: log-Euler for S, full-truncation Euler for the variance v
n_paths, n_steps = 200_000, 250
dt = T / n_steps
rng = np.random.default_rng(seed=40)
x, v = np.full(n_paths, np.log(S0)), np.full(n_paths, v0)
for _ in range(n_steps):
    z1 = rng.standard_normal(n_paths)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
    vp = np.maximum(v, 0)
    x += (r - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
    v += kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
ST = np.exp(x)
print("strike   MC price  std.err.  formula   implied vol")
for K in [80, 90, 100, 110, 120]:
    pay = np.exp(-r * T) * np.maximum(ST - K, 0)
    exact = heston_call(K)
    print(f"{K:5d} {pay.mean():10.4f} {pay.std()/np.sqrt(n_paths):8.4f} "
          f"{exact:9.4f} {implied_vol(exact, S0, K, r, T):10.4f}")
