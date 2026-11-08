import numpy as np
from mcfin_rng import MINSTD

def randint_ab(u, a, b):
    """Map uniforms u in [0, 1) to integers in {a, ..., b}."""
    return a + np.floor((b - a + 1) * np.asarray(u)).astype(int)

u = np.array(MINSTD(seed=31).sample(60_000))
dice = randint_ab(u, 1, 6)
values, counts = np.unique(dice, return_counts=True)
print("first 15 rolls:", dice[:15])
for v, c in zip(values, counts):
    print(f"  face {v}: {c:6d}  relative frequency {c / dice.size:.4f}   (1/6 = 0.1667)")
