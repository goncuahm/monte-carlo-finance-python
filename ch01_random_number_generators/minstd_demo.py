from mcfin_rng import MINSTD

g = MINSTD(seed=1)
print("first five integers:", [g.next_int() for _ in range(5)])

g = MINSTD(seed=1)
for _ in range(9999):                    # the 10,000th value of the sequence
    g.next_int()
print("x_10000 =", g.next_int(), "(published check value 1043618065)")

g = MINSTD(seed=2026)
print("five uniforms:", [round(u, 6) for u in g.sample(5)])
