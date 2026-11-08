import numpy as np

S0 = np.array([100.0, 100.0, 100.0])
vol = np.array([0.20, 0.30, 0.25])
w = np.array([1/3, 1/3, 1/3])                     # equally weighted basket
K, r, T, n = 100.0, 0.03, 1.0, 1_000_000

def basket_call(rho, seed=11):
    # equal correlation rho: Z_i = sqrt(rho) M + sqrt(1 - rho) e_i  (works for rho = 1 too)
    rng = np.random.default_rng(seed)
    M, e = rng.standard_normal((n, 1)), rng.standard_normal((n, 3))
    Z = np.sqrt(rho) * M + np.sqrt(1 - rho) * e
    ST = S0 * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)
    pay = np.exp(-r * T) * np.maximum(ST @ w - K, 0)
    return pay.mean(), pay.std(ddof=1) / np.sqrt(n)

print(" rho     price    std.error")
for rho in [0.0, 0.3, 0.6, 0.9, 1.0]:
    p, se = basket_call(rho)
    print(f"{rho:4.1f}   {p:7.4f}   {se:.4f}")
