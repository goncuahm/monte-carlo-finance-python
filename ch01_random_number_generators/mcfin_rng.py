"""mcfin_rng.py -- simple random number generators used in Chapter 1.

Teaching code only. For real work use numpy.random.default_rng.
"""

def lcg(seed, a, c, m, n):
    """Return the first n integers x_1, ..., x_n of the LCG
    x_{k+1} = (a * x_k + c) mod m, started from x_0 = seed."""
    x = seed
    out = []
    for _ in range(n):
        x = (a * x + c) % m
        out.append(x)
    return out


def lcg_uniform(seed, a, c, m, n):
    """Uniform numbers u_k = x_k / m in [0, 1) from the LCG."""
    return [x / m for x in lcg(seed, a, c, m, n)]


def period(seed, a, c, m):
    """Length of the cycle that the sequence started at `seed` falls into."""
    first_seen = {}
    x, k = seed, 0
    while x not in first_seen:
        first_seen[x] = k
        x = (a * x + c) % m
        k += 1
    return k - first_seen[x]


class MINSTD:
    """Park-Miller 'minimal standard' generator: a = 16807, c = 0, m = 2^31 - 1."""
    A, M = 16807, 2**31 - 1

    def __init__(self, seed=1):
        if not 0 < seed < self.M:
            raise ValueError("seed must be in 1, ..., 2^31 - 2")
        self.state = seed

    def next_int(self):
        self.state = (self.A * self.state) % self.M
        return self.state

    def random(self):
        """A uniform number in the open interval (0, 1)."""
        return self.next_int() / self.M

    def sample(self, n):
        return [self.random() for _ in range(n)]
