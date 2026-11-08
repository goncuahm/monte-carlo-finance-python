import numpy as np

# Two trades with the same counterparty: long 5-year and short 3-year EUR forward
S0, rd, rf, sigma, notional, M, N = 1.10, 0.04, 0.02, 0.10, 1e6, 60, 100_000
t = np.linspace(0, 5.0, M + 1)
rng = np.random.default_rng(seed=8)
logS = np.log(S0) + np.cumsum((rd - rf - 0.5 * sigma**2) * (5.0 / M)
                              + sigma * np.sqrt(5.0 / M) * rng.standard_normal((N, M)), axis=1)
S = np.hstack([np.full((N, 1), S0), np.exp(logS)])
def fwd(T, sign):
    K = S0 * np.exp((rd - rf) * T)
    tau = np.maximum(T - t, 0)
    v = sign * notional * (S * np.exp(-rf * tau) - K * np.exp(-rd * tau))
    return np.where(t <= T, v, 0.0)                    # trade has matured after T
V1, V2 = fwd(5.0, +1), fwd(3.0, -1)
lam, R = 0.02, 0.4
pd = np.exp(-lam * t[:-1]) - np.exp(-lam * t[1:])
def cva(EE):
    return (1 - R) * np.sum(np.exp(-rd * t[1:]) * EE[1:] * pd)
EE_sep = np.mean(np.maximum(V1, 0), axis=0) + np.mean(np.maximum(V2, 0), axis=0)
EE_net = np.mean(np.maximum(V1 + V2, 0), axis=0)
print(f"peak EE without netting {EE_sep.max():10,.0f}   CVA {cva(EE_sep):9,.0f}")
print(f"peak EE with netting    {EE_net.max():10,.0f}   CVA {cva(EE_net):9,.0f}")
print(f"netting reduces CVA by {100 * (1 - cva(EE_net) / cva(EE_sep)):.0f}%")
