import numpy as np

rng = np.random.default_rng(seed=5)
u = rng.random(200_000)

# Pareto with scale x_m = 1 and tail index alpha = 3:  X = x_m (1 - u)^(-1/alpha)
alpha = 3.0
pareto = (1 - u) ** (-1 / alpha)
# Weibull with scale 1 and shape k = 0.7:  X = (-ln(1 - u))^(1/k)
k = 0.7
weibull = (-np.log1p(-u)) ** (1 / k)
expo = -np.log1p(-u)                             # exponential(1), same uniforms

print("               mean   99% quantile   99.9% quantile   max")
for name, s in [("exponential", expo), ("Weibull k=0.7", weibull), ("Pareto a=3", pareto)]:
    q99, q999 = np.quantile(s, [0.99, 0.999])
    print(f"{name:14s} {s.mean():5.3f}   {q99:9.3f}     {q999:9.3f}     {s.max():8.2f}")
print(f"Pareto theory: mean {alpha/(alpha-1):.3f}, 99% quantile {0.01**(-1/alpha):.3f}")
