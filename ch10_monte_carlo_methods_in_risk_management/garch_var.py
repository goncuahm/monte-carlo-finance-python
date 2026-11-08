import numpy as np
from mcfin_risk import var_es

# GARCH(1,1): r_t = sigma_t z_t, sigma_t^2 = w + a r_{t-1}^2 + b sigma_{t-1}^2
w, a, b = 1e-6, 0.08, 0.90
long_run = np.sqrt(w / (1 - a - b))                     # long-run daily volatility
print(f"long-run daily vol {100*long_run:.3f}%  (annual {100*long_run*np.sqrt(252):.1f}%)\n")
N, H = 200_000, 10
print("current daily vol   1-day 99% VaR   10-day 99% VaR   sqrt(10) x 1-day   ES 10-day")
for sig0 in [0.5 * long_run, long_run, 2.5 * long_run]:
    rng = np.random.default_rng(seed=5)
    s2, r_prev = np.full(N, sig0**2), np.zeros(N)
    total = np.zeros(N)
    for k in range(H):
        if k > 0:
            s2 = w + a * r_prev**2 + b * s2
        r_prev = np.sqrt(s2) * rng.standard_normal(N)
        total += r_prev
        if k == 0:
            v1, _ = var_es(-r_prev, 0.99)
    v10, e10 = var_es(-total, 0.99)
    print(f"{100*sig0:12.3f}% {100*v1:14.2f}% {100*v10:15.2f}% {100*np.sqrt(H)*v1:16.2f}%"
          f" {100*e10:11.2f}%")
