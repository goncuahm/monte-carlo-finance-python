import numpy as np

# 'Historical' daily returns of a stock (synthetic data for illustration, 252 trading days)
hist = np.random.default_rng(0).standard_t(df=4, size=252) * 0.012 + 0.0004

rng = np.random.default_rng(seed=123)
n_sims, horizon = 20_000, 21                        # one month = 21 trading days
idx = rng.integers(0, hist.size, size=(n_sims, horizon))   # random days, with replacement
monthly = np.prod(1 + hist[idx], axis=1) - 1        # compound the resampled daily returns

print(f"mean monthly return     {monthly.mean():+.4f}")
print(f"std. dev. monthly       {monthly.std(ddof=1):.4f}")
print(f"P(loss > 5%)            {np.mean(monthly < -0.05):.4f}")
print(f"5% quantile (95% VaR)   {np.quantile(monthly, 0.05):+.4f}")
