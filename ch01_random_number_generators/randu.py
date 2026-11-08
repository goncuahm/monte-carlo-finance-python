import numpy as np
import matplotlib.pyplot as plt

a, m = 65539, 2**31                          # RANDU (IBM, 1960s)
n = 30_000
x = np.empty(n, dtype=np.int64)
x[0] = 1
for k in range(1, n):
    x[k] = (a * x[k - 1]) % m

# every triple satisfies x_{k+2} - 6 x_{k+1} + 9 x_k = 0 (mod 2^31)
check = (x[2:] - 6 * x[1:-1] + 9 * x[:-2]) % m
print("all triples on the planes:", np.all(check == 0))

u = x / m
s = 9 * u[:-2] - 6 * u[1:-1] + u[2:]         # an integer for every triple
print("distinct values of 9u_k - 6u_(k+1) + u_(k+2):", np.unique(np.round(s)).astype(int))

fig = plt.figure(figsize=(7.2, 3.4))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(u[:-2:3], u[1:-1:3], u[2::3], s=0.6, color="#2a5d8f")
ax1.set_title("RANDU triples")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.scatter(u[:-2:3], u[1:-1:3], u[2::3], s=0.6, color="#b5523b")
ax2.view_init(elev=-31, azim=56)              # look along the planes
ax2.set_title("same points, rotated")
ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1]); ax1.set_zticks([0, 1])
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
plt.tight_layout()
plt.show()
