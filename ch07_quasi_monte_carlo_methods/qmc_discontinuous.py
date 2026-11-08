import numpy as np
from mcfin_qmc import sobol_normals, path_matrix

# Variance ratio MC / RQMC (Sobol + bridge, N = 2^12, 32 scramblings) for three payoffs
S0, K, B, r, sigma, T, n, m, R = 100.0, 100.0, 85.0, 0.05, 0.3, 1.0, 64, 12, 32
t = T * np.arange(1, n + 1) / n
A = path_matrix(n, T, "bridge")
def payoffs(Z):
    S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * Z @ A.T)
    d = np.exp(-r * T)
    return {"Asian call": d * np.maximum(S.mean(axis=1) - K, 0),
            "digital call": d * (S[:, -1] > K),
            "down-and-out call": d * np.maximum(S[:, -1] - K, 0) * (S.min(axis=1) > B)}
q = {k: [] for k in payoffs(np.zeros((1, n)))}
mc = {k: [] for k in q}
for s in range(R):
    for k, v in payoffs(sobol_normals(m, n, s)).items():
        q[k].append(v.mean())
    for k, v in payoffs(np.random.default_rng(s).standard_normal((2**m, n))).items():
        mc[k].append(v.mean())
print(f"{'payoff':20s} {'price':>8s} {'MC std':>9s} {'RQMC std':>9s} {'ratio':>7s}")
for k in q:
    a, b = np.std(mc[k], ddof=1), np.std(q[k], ddof=1)
    print(f"{k:20s} {np.mean(q[k]):8.4f} {a:9.5f} {b:9.5f} {(a / b)**2:7.1f}")
