import numpy as np
from mcfin_path import mc_price, bs_call
from scipy.optimize import brentq

# Heston model of Chapter 4 versus Black-Scholes calibrated to the ATM Heston price
S0, K, B, r, T, n_steps, n_paths = 100.0, 100.0, 85.0, 0.03, 1.0, 252, 400_000
v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.6, -0.7
dt = T / n_steps

def simulate(model, seed=60):
    rng = np.random.default_rng(seed)
    x, v = np.full(n_paths, np.log(S0)), np.full(n_paths, v0)
    alive, s_sum = np.ones(n_paths, dtype=bool), np.zeros(n_paths)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        if model == "heston":
            z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
            vp = np.maximum(v, 0)
            x += (r - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
            v += kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
        else:
            x += (r - 0.5 * sig_bs**2) * dt + sig_bs * np.sqrt(dt) * z1
        S = np.exp(x)
        alive &= S > B
        s_sum += S
    disc = np.exp(-r * T)
    return {"European call": disc * np.maximum(S - K, 0),
            "down-and-out call, B = 85": disc * np.maximum(S - K, 0) * alive,
            "down-and-in call, B = 85": disc * np.maximum(S - K, 0) * ~alive,
            "Asian call (daily)": disc * np.maximum(s_sum / n_steps - K, 0)}

sig_bs = 0.0
hes = simulate("heston")
atm = hes["European call"].mean()
sig_bs = brentq(lambda s: bs_call(S0, K, r, s, T) - atm, 0.01, 1.0)
print(f"Black-Scholes volatility matching the Heston ATM call: {sig_bs:.4f}\n")
bs = simulate("bs")
print(f"{'product':28s} {'Heston':>8s} {'BS':>8s} {'diff %':>8s}")
for name in hes:
    a, b = hes[name].mean(), bs[name].mean()
    print(f"{name:28s} {a:8.4f} {b:8.4f} {100*(a-b)/b:+8.1f}")
