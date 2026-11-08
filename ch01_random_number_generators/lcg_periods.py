from mcfin_rng import period

m = 16
cases = [(5, 3, "full period"),
         (5, 4, "c and m share the factor 2"),
         (3, 3, "a - 1 = 2 is not divisible by 4"),
         (1, 3, "a = 1: x_k = x_0 + k c"),
         (5, 0, "multiplicative, c = 0")]
for a, c, comment in cases:
    lengths = sorted({period(s, a, c, m) for s in range(m)})
    print(f"a = {a}, c = {c}:  cycle lengths {lengths}   ({comment})")
