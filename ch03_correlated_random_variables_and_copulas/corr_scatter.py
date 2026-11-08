import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)
Z = rng.standard_normal((1_000, 2))
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0))
for axis, rho in zip(ax, [-0.8, 0.0, 0.5, 0.95]):
    x2 = rho * Z[:, 0] + np.sqrt(1 - rho**2) * Z[:, 1]
    axis.scatter(Z[:, 0], x2, s=2, color="#2a5d8f")
    axis.set_title(f"rho = {rho}"); axis.set_xlim(-4, 4); axis.set_ylim(-4, 4)
    axis.set_aspect("equal"); axis.set_xticks([-3, 0, 3]); axis.set_yticks([-3, 0, 3])
plt.tight_layout()
plt.show()
