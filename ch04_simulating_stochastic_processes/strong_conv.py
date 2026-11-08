import numpy as np
import matplotlib.pyplot as plt

# GBM dS = mu S dt + sigma S dW: Euler and Milstein against the exact solution
S0, mu, sigma, T, n_paths = 100.0, 0.05, 0.4, 1.0, 50_000
rng = np.random.default_rng(seed=3)
n_fine = 2**10
dW_fine = np.sqrt(T / n_fine) * rng.standard_normal((n_paths, n_fine))
S_exact = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * dW_fine.sum(axis=1))

steps, e_eul, e_mil = [2**k for k in range(1, 9)], [], []
for n in steps:
    dt = T / n
    dW = dW_fine.reshape(n_paths, n, -1).sum(axis=2)   # same Brownian path, coarser
    Se, Sm = np.full(n_paths, S0), np.full(n_paths, S0)
    for k in range(n):
        Se = Se + mu * Se * dt + sigma * Se * dW[:, k]
        Sm = Sm + mu * Sm * dt + sigma * Sm * dW[:, k] \
                + 0.5 * sigma**2 * Sm * (dW[:, k]**2 - dt)
    e_eul.append(np.mean(np.abs(Se - S_exact)))
    e_mil.append(np.mean(np.abs(Sm - S_exact)))

print(" steps   Euler error   Milstein error")
for n, a, b in zip(steps, e_eul, e_mil):
    print(f"{n:6d} {a:12.4f} {b:14.4f}")
h = np.log(T / np.array(steps))
print(f"estimated order: Euler {np.polyfit(h, np.log(e_eul), 1)[0]:.2f}, "
      f"Milstein {np.polyfit(h, np.log(e_mil), 1)[0]:.2f}")

fig, ax = plt.subplots(figsize=(5.6, 2.9))
ax.loglog(T / np.array(steps), e_eul, "o-", label="Euler")
ax.loglog(T / np.array(steps), e_mil, "s-", label="Milstein")
ax.set_xlabel("time step h"); ax.set_ylabel("mean absolute error"); ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
