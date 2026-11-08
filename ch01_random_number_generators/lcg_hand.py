from mcfin_rng import lcg

m, a, c, seed = 16, 5, 3, 7
xs = lcg(seed, a, c, m, 18)
print(" k   x_k   u_k = x_k/16")
print(f" 0   {seed:3d}   {seed/m:.4f}")
for k, x in enumerate(xs, start=1):
    print(f"{k:2d}   {x:3d}   {x/m:.4f}")
