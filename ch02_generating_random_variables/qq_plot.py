import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(9)
z = rng.standard_normal(2_000)
t4 = rng.standard_t(4, 2_000) * np.sqrt(2 / 4)     # Student t(4) rescaled to variance 1

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
for axis, data, title in [(ax[0], z, "normal sample"), (ax[1], t4, "Student t(4), variance 1")]:
    (theo, samp), _ = stats.probplot(data, dist="norm")
    axis.scatter(theo, samp, s=4, color="#2a5d8f")
    axis.plot([-4, 4], [-4, 4], color="black", lw=1)
    axis.set_xlim(-4, 4); axis.set_ylim(-7, 7)
    axis.set_xlabel("normal quantiles"); axis.set_ylabel("sample quantiles"); axis.set_title(title)
plt.tight_layout()
plt.show()
