import numpy as np

# CIR short rate dr = kappa (theta - r) dt + sigma sqrt(r) dW
kappa, theta, sigma, r0, T = 0.5, 0.04, 0.15, 0.02, 5.0
print(f"Feller condition 2 kappa theta >= sigma^2: {2*kappa*theta:.4f} vs {sigma**2:.4f}")
h = np.sqrt(kappa**2 + 2 * sigma**2)
den = 2 * h + (kappa + h) * (np.exp(h * T) - 1)
B = 2 * (np.exp(h * T) - 1) / den
A = (2 * h * np.exp((kappa + h) * T / 2) / den)**(2 * kappa * theta / sigma**2)
print(f"closed-form bond price {A*np.exp(-B*r0):.5f}\n")

def bond_mc(n_steps, scheme, n_paths=100_000, seed=9):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    r, integral = np.full(n_paths, r0), np.zeros(n_paths)
    c = sigma**2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
    df = 4 * kappa * theta / sigma**2
    for k in range(n_steps):
        r_old = r
        if scheme == "exact":                       # noncentral chi-square transition
            r = c * rng.noncentral_chisquare(df, r * np.exp(-kappa * dt) / c)
        else:                                       # Euler with full truncation
            rp = np.maximum(r, 0)
            r = r + kappa * (theta - rp) * dt \
                  + sigma * np.sqrt(rp * dt) * rng.standard_normal(n_paths)
        integral += 0.5 * (np.maximum(r_old, 0) + np.maximum(r, 0)) * dt
    disc = np.exp(-integral)
    return disc.mean(), disc.std() / np.sqrt(n_paths), np.mean(r < 0)

print(" scheme       steps   bond price  std.err.  P(r<0)")
for scheme, n in [("Euler", 10), ("Euler", 50), ("Euler", 250), ("exact", 250)]:
    p, se, neg = bond_mc(n, scheme)
    print(f" {scheme:8s} {n:8d} {p:12.5f} {se:9.5f} {neg:7.4f}")
