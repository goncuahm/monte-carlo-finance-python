import numpy as np

# 3-year autocallable note on one stock, notional 100, quarterly observations.
# At each date: if S >= S0 the note is redeemed at 100 + 2.5 per elapsed quarter.
# At maturity, if never called: 100 if S(T) >= 60, otherwise 100 * S(T)/S0.
S0, r, sigma, T, n_obs, n_paths = 100.0, 0.03, 0.25, 3.0, 12, 1_000_000
coupon, ki = 2.5, 60.0
steps_per_obs = 21                                   # daily steps between observations
dt = T / (n_obs * steps_per_obs)
rng = np.random.default_rng(seed=10)
logS = np.full(n_paths, np.log(S0))
value = np.zeros(n_paths)
alive = np.ones(n_paths, dtype=bool)
call_prob = []
for j in range(1, n_obs + 1):
    z = rng.standard_normal((steps_per_obs, n_paths))
    logS += ((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z).sum(axis=0)
    t, S = j * T / n_obs, np.exp(logS)
    called = alive & (S >= S0)
    value[called] = np.exp(-r * t) * (100 + coupon * j)
    call_prob.append(called.mean())
    alive &= ~called
redeem = np.where(S >= ki, 100.0, 100.0 * S / S0)    # final redemption if never called
value[alive] = np.exp(-r * T) * redeem[alive]

print(f"note value {value.mean():.3f} +/- {1.96*value.std()/np.sqrt(n_paths):.3f} per 100")
print("probability of being called at quarter:",
      " ".join(f"{p:.3f}" for p in call_prob[:6]), "...")
print(f"P(called at some date)        {sum(call_prob):.4f}")
print(f"P(not called, capital loss)   {np.mean(alive & (S < ki)):.4f}")
print(f"expected life (years)         "
      f"{sum(p * (j+1) * T / n_obs for j, p in enumerate(call_prob)) + alive.mean() * T:.3f}")
