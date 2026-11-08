import numpy as np

# Fill in W(t) between W(s) = a and W(u) = b, for s < t < u
def bridge(rng, s, a, u, b, t):
    mean = a + (t - s) / (u - s) * (b - a)
    std = np.sqrt((t - s) * (u - t) / (u - s))
    return mean + std * rng.standard_normal(np.shape(b))

rng = np.random.default_rng(seed=8)
n = 200_000
W1 = rng.standard_normal(n)                   # first the end point W(1)
W_half = bridge(rng, 0, 0.0, 1, W1, 0.5)      # then the midpoint
W_q = bridge(rng, 0, 0.0, 0.5, W_half, 0.25)  # then the quarter points
W_3q = bridge(rng, 0.5, W_half, 1, W1, 0.75)
for name, x, v in [("W(0.25)", W_q, 0.25), ("W(0.5)", W_half, 0.5),
                   ("W(0.75)", W_3q, 0.75), ("W(1)", W1, 1.0)]:
    print(f"Var {name:8s} = {x.var():.4f}   (theory {v})")
print(f"Cov W(0.25),W(0.75) = {np.cov(W_q, W_3q)[0, 1]:.4f}   (theory 0.25)")
