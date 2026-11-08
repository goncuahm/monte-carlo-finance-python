import numpy as np
from mcfin_greeks import mean_se

# Arithmetic Asian call, 12 monthly dates: delta and vega by three methods
S0, K, r, sigma, T, n, N = 100.0, 100.0, 0.05, 0.3, 1.0, 12, 200_000
h = T / n
t = h * np.arange(1, n + 1)
rng = np.random.default_rng(seed=7)
Z = rng.standard_normal((N, n))
W = np.sqrt(h) * np.cumsum(Z, axis=1)
disc = np.exp(-r * T)
def paths(s0, sig):
    return s0 * np.exp((r - 0.5 * sig**2) * t + sig * W)
S = paths(S0, sigma)
A = S.mean(axis=1)
pay = disc * np.maximum(A - K, 0)
itm = A > K
# pathwise: dA/dS0 = A/S0, dA/dsigma = mean of S_i (W_i - sigma t_i)
d_pw = disc * itm * A / S0
v_pw = disc * itm * (S * (W - sigma * t)).mean(axis=1)
# likelihood ratio: only the first increment depends on S0; for sigma use all steps
d_lr = pay * Z[:, 0] / (S0 * sigma * np.sqrt(h))
v_lr = pay * np.sum((Z**2 - 1) / sigma - Z * np.sqrt(h), axis=1)
# finite differences with common random numbers
fd = lambda s0, sig: disc * np.maximum(paths(s0, sig).mean(axis=1) - K, 0)
d_fd = (fd(S0 + 0.5, sigma) - fd(S0 - 0.5, sigma)) / 1.0
v_fd = (fd(S0, sigma + 0.005) - fd(S0, sigma - 0.005)) / 0.01
print(f"Asian call price {pay.mean():.4f}\n")
print("method              delta (s.e.)          vega (s.e.)")
for name, d, v in [("pathwise", d_pw, v_pw), ("likelihood ratio", d_lr, v_lr),
                   ("FD with CRN", d_fd, v_fd)]:
    (md, sd), (mv, sv) = mean_se(d), mean_se(v)
    print(f"{name:18s} {md:8.4f} ({sd:.4f})   {mv:9.4f} ({sv:.4f})")
