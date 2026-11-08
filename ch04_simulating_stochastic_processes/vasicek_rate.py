import numpy as np

# Vasicek short rate dr = kappa (theta - r) dt + sigma dW, simulated exactly
kappa, theta, sigma, r0, T, n_steps, n_paths = 0.5, 0.04, 0.01, 0.02, 5.0, 500, 100_000
dt = T / n_steps
e = np.exp(-kappa * dt)
sd = sigma * np.sqrt((1 - e**2) / (2 * kappa))      # exact conditional std. dev.
rng = np.random.default_rng(seed=21)
r = np.full(n_paths, r0)
integral = 0.5 * r * dt                              # trapezoid rule for int r dt
for k in range(n_steps):
    r = theta + (r - theta) * e + sd * rng.standard_normal(n_paths)
    integral += (r if k < n_steps - 1 else 0.5 * r) * dt
m_th = theta + (r0 - theta) * np.exp(-kappa * T)
v_th = sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * T))
print(f"E[r(5)]  {r.mean():.5f}   theory {m_th:.5f}")
print(f"sd[r(5)] {r.std():.5f}   theory {np.sqrt(v_th):.5f}")
print(f"P(r(5) < 0) = {np.mean(r < 0):.4f}")

# Zero-coupon bond P(0,T) = E[exp(-int_0^T r dt)] against the closed form
disc = np.exp(-integral)
B = (1 - np.exp(-kappa * T)) / kappa
A = np.exp((theta - sigma**2 / (2 * kappa**2)) * (B - T) - sigma**2 * B**2 / (4 * kappa))
print(f"bond price MC {disc.mean():.5f} +/- {1.96*disc.std()/np.sqrt(n_paths):.5f}"
      f"   formula {A*np.exp(-B*r0):.5f}")
