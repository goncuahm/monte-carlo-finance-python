import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)
y = rng.random(1_500)                      # proposals from U(0,1)
v = 1.5 * rng.random(1_500)                # heights uniform on [0, c g(y)] = [0, 1.5]
ok = v <= 6 * y * (1 - y)

fig, ax = plt.subplots(figsize=(5.2, 2.9))
ax.scatter(y[ok], v[ok], s=4, color="#2a5d8f", label="accepted")
ax.scatter(y[~ok], v[~ok], s=4, color="#c9962b", label="rejected")
t = np.linspace(0, 1, 200)
ax.plot(t, 6 * t * (1 - t), color="black", label="f(x) = 6x(1-x)")
ax.axhline(1.5, color="black", ls="--", lw=1, label="c g(x) = 1.5")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=7, frameon=False)
ax.set_title(f"acceptance rate {ok.mean():.3f}")
plt.tight_layout()
plt.show()
