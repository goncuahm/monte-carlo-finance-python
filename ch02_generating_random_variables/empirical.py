import numpy as np

# Daily returns observed on 10 days (in %), a tiny 'historical' sample
hist = np.array([-2.1, -0.8, -0.3, 0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 2.4])
rng = np.random.default_rng(seed=4)
u = rng.random(100_000)

# (a) discrete empirical distribution: each observation has probability 1/10
idx = np.floor(u * hist.size).astype(int)
draws_a = hist[idx]

# (b) smoothed: interpolate the empirical quantile function between observations
grid = (np.arange(hist.size) + 0.5) / hist.size  # plotting positions
draws_b = np.interp(u, grid, np.sort(hist))

print(f"historical mean {hist.mean():+.3f}, std {hist.std(ddof=1):.3f}")
print(f"(a) resampled:  mean {draws_a.mean():+.3f}, std {draws_a.std(ddof=1):.3f}, "
      f"distinct values {np.unique(draws_a).size}")
print(f"(b) smoothed:   mean {draws_b.mean():+.3f}, std {draws_b.std(ddof=1):.3f}, "
      f"distinct values {np.unique(draws_b).size}")
print(f"5% quantile:  (a) {np.quantile(draws_a, 0.05):+.3f}   (b) {np.quantile(draws_b, 0.05):+.3f}")
