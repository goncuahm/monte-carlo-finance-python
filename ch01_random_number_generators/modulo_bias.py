import numpy as np

# A source of uniform integers 0..7 (3 random bits). We want 0, 1 or 2.
source = np.arange(8)
print("x mod 3 for x = 0..7:", source % 3)
print("P(0), P(1), P(2) =", [float(np.mean(source % 3 == j)) for j in range(3)], " (should be 1/3 each)")

# Rejection: discard x >= 6 (the largest multiple of 3 not exceeding 8), then take x mod 3
rng = np.random.default_rng(1)
x = rng.integers(0, 8, size=300_000)
kept = x[x < 6] % 3
print(f"accepted fraction: {kept.size / x.size:.3f}  (theory 6/8 = 0.75)")
print("frequencies after rejection:", np.round(np.bincount(kept) / kept.size, 4))
