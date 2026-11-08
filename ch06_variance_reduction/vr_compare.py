import numpy as np
from scipy.stats import norm
from mcfin_vr import summary, control_variate, Timer

# All methods on the arithmetic Asian call (monthly averaging), same budget of paths
S0, K, r, sigma, T, n, N = 100.0, 100.0, 0.05, 0.30, 1.0, 12, 200_000
dt, disc = T / n, np.exp(-r * T)
m = np.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
d1 = (m - np.log(K) + v) / np.sqrt(v)
geo = disc * (np.exp(m + v / 2) * norm.cdf(d1) - K * norm.cdf(d1 - np.sqrt(v)))

def payoffs(Z):
    logS = np.log(S0) + np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    return (disc * np.maximum(np.exp(logS).mean(axis=1) - K, 0),
            disc * np.maximum(np.exp(logS.mean(axis=1)) - K, 0))

def run(method, rng):
    if method == "plain":
        Y, _ = payoffs(rng.standard_normal((N, n))); return summary(Y)
    if method == "antithetic":
        Z = rng.standard_normal((N // 2, n))
        Y = 0.5 * (payoffs(Z)[0] + payoffs(-Z)[0]); return summary(Y, n_payoffs=N)
    if method == "control":
        Y, G = payoffs(rng.standard_normal((N, n)))
        return summary(control_variate(Y, G, geo)[0])
    if method == "antithetic + control":
        Z = rng.standard_normal((N // 2, n))
        (Y1, G1), (Y2, G2) = payoffs(Z), payoffs(-Z)
        Ycv, _ = control_variate(0.5 * (Y1 + Y2), 0.5 * (G1 + G2), geo)
        return summary(Ycv, n_payoffs=N)

print(f"{'method':22s} {'price':>8s} {'std.err.':>9s} {'time':>7s} {'efficiency gain':>16s}")
for meth in ["plain", "antithetic", "control", "antithetic + control"]:
    with Timer() as t:
        p, se, _ = run(meth, np.random.default_rng(seed=7))
    if meth == "plain":
        cost0 = se**2 * t.s
    print(f"{meth:22s} {p:8.4f} {se:9.5f} {t.s:7.3f} {cost0 / (se**2 * t.s):16.1f}")
