import numpy as np
from mcfin_sp import brownian_paths

rng = np.random.default_rng(seed=1)
t, W = brownian_paths(rng, T=1.0, n_steps=100, n_paths=100_000)
i, j = 25, 75                                         # s = 0.25, t = 0.75
print(f"E[W(0.75)]          = {W[:, j].mean():+.4f}   (theory 0)")
print(f"Var[W(0.75)]        = {W[:, j].var():.4f}    (theory 0.75)")
print(f"Cov[W(0.25),W(0.75)] = {np.cov(W[:, i], W[:, j])[0, 1]:.4f}    (theory 0.25)")
inc1, inc2 = W[:, i], W[:, j] - W[:, i]              # increments on disjoint intervals
print(f"corr of increments  = {np.corrcoef(inc1, inc2)[0, 1]:+.4f}   (theory 0)")

# Quadratic variation versus total variation of one path on finer grids
rng = np.random.default_rng(seed=2)
_, w = brownian_paths(rng, T=1.0, n_steps=2**16, n_paths=1)
print("\n   steps   sum |dW|   sum dW^2")
for n in [2**4, 2**8, 2**12, 2**16]:
    dW = np.diff(w[0, ::2**16 // n])
    print(f"{n:8d} {np.abs(dW).sum():10.3f} {np.sum(dW**2):10.4f}")
