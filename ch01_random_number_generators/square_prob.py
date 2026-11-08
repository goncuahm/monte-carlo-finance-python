import numpy as np

rng = np.random.default_rng(seed=42)
n = 1_000_000
u, v = rng.random(n), rng.random(n)

# Two independent U(0,1) 'returns scores'. Probability both exceed 0.8:
print(f"P(U > 0.8 and V > 0.8):  MC {np.mean((u > 0.8) & (v > 0.8)):.5f}   exact {0.2*0.2:.5f}")
# Probability their sum exceeds 1.5 (triangle of area 1/8):
print(f"P(U + V > 1.5):          MC {np.mean(u + v > 1.5):.5f}   exact {0.125:.5f}")
# Expected value of the larger of the two: E[max(U, V)] = 2/3
print(f"E[max(U, V)]:            MC {np.mean(np.maximum(u, v)):.5f}   exact {2/3:.5f}")
