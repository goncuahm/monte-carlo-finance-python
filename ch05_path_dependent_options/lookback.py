import numpy as np
from mcfin_path import mc_price, lookback_call

S0, r, sigma, T, n_paths = 100.0, 0.05, 0.30, 1.0, 400_000
print(f"continuous lookback formula {lookback_call(S0, r, sigma, T):.4f}\n")
print("   n   grid minimum   s.e.   bridge minimum   s.e.")
for n in [4, 12, 52, 252]:
    rng = np.random.default_rng(seed=5)
    h = T / n
    x = np.full(n_paths, np.log(S0))
    m_grid, m_bridge = x.copy(), x.copy()
    for _ in range(n):
        x_new = x + (r - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * rng.standard_normal(n_paths)
        # minimum of the Brownian bridge between x and x_new (exact sampling)
        U = rng.random(n_paths)
        low = 0.5 * (x + x_new - np.sqrt((x_new - x)**2 - 2 * sigma**2 * h * np.log(U)))
        m_grid = np.minimum(m_grid, x_new)
        m_bridge = np.minimum(m_bridge, low)
        x = x_new
    ST = np.exp(x)
    (p1, s1) = mc_price(np.exp(-r * T) * (ST - np.exp(m_grid)))
    (p2, s2) = mc_price(np.exp(-r * T) * (ST - np.exp(m_bridge)))
    print(f"{n:5d} {p1:11.4f} {s1:8.4f} {p2:13.4f} {s2:8.4f}")
