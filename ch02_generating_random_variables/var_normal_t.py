import numpy as np
from scipy import stats

rng = np.random.default_rng(seed=99)
n, vol, value = 1_000_000, 0.015, 1_000_000      # daily vol 1.5%, position EUR 1m
nu = 4
returns = {"normal": vol * rng.standard_normal(n),
           "Student t(4)": vol * np.sqrt((nu - 2) / nu) * rng.standard_t(nu, n)}   # same variance

print("                std      99% VaR     99.9% VaR    99% ES")
for name, x in returns.items():
    loss = -value * x
    v99, v999 = np.quantile(loss, [0.99, 0.999])
    es99 = loss[loss >= v99].mean()
    print(f"{name:13s} {x.std():.4f}   {v99:9,.0f}   {v999:9,.0f}   {es99:9,.0f}")
print(f"normal formula: 99% VaR = {value * vol * stats.norm.ppf(0.99):,.0f}")
