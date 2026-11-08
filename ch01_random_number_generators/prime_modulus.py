from mcfin_rng import period

m = 31                                   # a prime modulus
for a in [2, 3, 5, 7, 11, 12]:
    print(f"a = {a:2d}: period from seed 1 = {period(1, a, 0, m)}")
