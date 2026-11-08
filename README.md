# Monte Carlo Methods in Quantitative Finance with Python

Python code for the textbook **Monte Carlo Methods in Quantitative Finance with Python**
by **Ahmet Göncü, PhD** (AGPublications, 2026).

Every listing printed in the book is here, organised by chapter. Each program reproduces
the output and figures shown in the book.

## Getting started

1. Install Python 3 and the three libraries used throughout the book:
   ```
   pip install -r requirements.txt
   ```
2. Download this repository with the green **Code → Download ZIP** button (or `git clone`).
3. Open a chapter folder and run a listing, for example:
   ```
   cd ch01_random_number_generators
   python lcg_hand.py
   ```

Each chapter folder contains the helper module (`mcfin_*.py`) that its listings import,
so every folder works on its own. Programs that draw figures open them with `plt.show()`;
replace it with `plt.savefig("figure.pdf")` to save the figure instead.

All programs set a random seed, so results are reproducible. Very rarely, a different NumPy
version can change the last digits of an output; timings depend on your computer.

## Market data

`data/download_data.py` downloads the daily prices used in the real-data examples from
Yahoo Finance (requires `pip install yfinance pandas`).

## Contents

### Chapter 1: Random Number Generators
Folder: [`ch01_random_number_generators/`](ch01_random_number_generators/)

| Listing | File | Description |
|---|---|---|
| 1.1 | `mcfin_rng.py` | Our module of simple random number generators |
| 1.2 | `lcg_hand.py` | The LCG of Example 1.5 |
| 1.3 | `lcg_periods.py` | Cycle lengths for every seed, m=16 |
| 1.4 | `prime_modulus.py` | Periods of Lehmer generators with m=31 |
| 1.5 | `minstd_demo.py` | The Park-Miller minimal standard generator |
| 1.6 | `uniform_moments.py` | Sample moments of MINSTD uniforms |
| 1.7 | `hist_uniform.py` | Histograms of uniform numbers |
| 1.8 | `chi_square.py` | Chi-square and Kolmogorov-Smirnov tests |
| 1.9 | `serial_corr.py` | Serial correlation of two generators |
| 1.10 | `square_prob.py` | Probabilities on the unit square by simulation |
| 1.11 | `pi_estimate.py` | Monte Carlo estimate of pi |
| 1.12 | `unit_square.py` | Consecutive pairs on the unit square |
| 1.13 | `randu.py` | The fifteen planes of RANDU |
| 1.14 | `numpy_rng.py` | Modern generators in Python and NumPy |
| 1.15 | `int_from_uniform.py` | Random integers from uniform numbers |
| 1.16 | `modulo_bias.py` | Modulo bias and its removal by rejection |
| 1.17 | `numpy_integers.py` | Random integers, sampling and permutations |
| 1.18 | `bootstrap.py` | Monthly return distribution by bootstrapping |
| 1.19 | `binomial_mc.py` | Monte Carlo price of a call in the binomial model |

### Chapter 2: Generating Random Variables
Folder: [`ch02_generating_random_variables/`](ch02_generating_random_variables/)

| Listing | File | Description |
|---|---|---|
| 2.1 | `mcfin_rv.py` | Our module for generating random variables |
| 2.2 | `exp_inverse.py` | Exponential variables by inversion |
| 2.3 | `inverse_figure.py` | The inverse transform, illustrated |
| 2.4 | `heavy_tails.py` | Exponential, Weibull and Pareto tails |
| 2.5 | `discrete_dists.py` | Discrete distributions by inversion |
| 2.6 | `poisson_process.py` | Poisson counts from exponential waiting times |
| 2.7 | `empirical.py` | Discrete and smoothed empirical distributions |
| 2.8 | `accept_reject.py` | Acceptance-rejection in practice |
| 2.9 | `ar_figure.py` | The geometry of acceptance-rejection |
| 2.10 | `normal_methods.py` | Four ways to generate standard normals |
| 2.11 | `qq_plot.py` | Normal Q-Q plots |
| 2.12 | `normality_tests.py` | The Jarque-Bera test |
| 2.13 | `lognormal.py` | Terminal stock prices in the Black-Scholes model |
| 2.14 | `chi2_t.py` | Chi-square and Student-t from normals |
| 2.15 | `bs_mc.py` | Monte Carlo price of a European call |
| 2.16 | `bs_convergence.py` | Convergence of the Monte Carlo price |
| 2.17 | `default_times.py` | Simulated default times and a risky bond |
| 2.18 | `jumps.py` | Poisson numbers of jumps and their effect |
| 2.19 | `var_normal_t.py` | Value-at-Risk and Expected Shortfall under two models |
| 2.20 | `numpy_dists.py` | Common distributions in NumPy |

### Chapter 3: Correlated Random Variables and Copulas
Folder: [`ch03_correlated_random_variables_and_copulas/`](ch03_correlated_random_variables_and_copulas/)

| Listing | File | Description |
|---|---|---|
| 3.1 | `mcfin_mv.py` | Our module for correlated variables and copulas |
| 3.2 | `bivariate.py` | Two correlated normals |
| 3.3 | `corr_scatter.py` | Scatter plots for different correlations |
| 3.4 | `cholesky_hand.py` | The Cholesky factor of Example 3.5 |
| 3.5 | `mvn_assets.py` | Correlated returns of four assets |
| 3.6 | `not_psd.py` | Detecting and repairing an invalid correlation matrix |
| 3.7 | `eigen_sampling.py` | Sampling with the eigen-decomposition |
| 3.8 | `basket_option.py` | Price of a basket call for different correlations |
| 3.9 | `margrabe.py` | Exchange option: Monte Carlo against Margrabe's formula |
| 3.10 | `portfolio_var.py` | Portfolio Value-at-Risk and correlation |
| 3.11 | `corr_limits.py` | Two failures of linear correlation |
| 3.12 | `rank_corr.py` | Pearson, Spearman and Kendall correlations |
| 3.13 | `copula_scatter.py` | Gaussian and t copulas |
| 3.14 | `tail_dependence.py` | Joint exceedance probabilities |
| 3.15 | `copula_marginals.py` | A Gaussian copula with Student-t marginals |
| 3.16 | `first_to_default.py` | Correlated default times of five firms |
| 3.17 | `vasicek.py` | Loss distribution of a portfolio of 100 loans |
| 3.18 | `loss_hist.py` | Distribution of the number of defaults |
| 3.19 | `t_copula_credit.py` | Gaussian versus t copula in a credit portfolio |

### Chapter 4: Simulating Stochastic Processes
Folder: [`ch04_simulating_stochastic_processes/`](ch04_simulating_stochastic_processes/)

| Listing | File | Description |
|---|---|---|
| 4.1 | `mcfin_sp.py` | Our module for stochastic processes |
| 4.2 | `bm_paths.py` | Ten paths of Brownian motion |
| 4.3 | `bm_properties.py` | Checking the properties of simulated Brownian motion |
| 4.4 | `brownian_bridge.py` | Building a path from its end point: the Brownian bridge |
| 4.5 | `gbm_fan.py` | Paths and quantile bands of geometric Brownian motion |
| 4.6 | `gbm_check.py` | Checking the distribution of simulated GBM |
| 4.7 | `strong_conv.py` | Strong convergence of the Euler and Milstein schemes |
| 4.8 | `weak_error.py` | Weak error of the Euler scheme for a call option |
| 4.9 | `vasicek_rate.py` | Exact simulation of the Vasicek model and a bond price |
| 4.10 | `cir_rate.py` | CIR model: Euler with full truncation versus exact simulation |
| 4.11 | `rate_paths.py` | Paths of the Vasicek and CIR models |
| 4.12 | `heston.py` | Heston model: Monte Carlo against the semi-analytic formula |
| 4.13 | `poisson_events.py` | Event times and counts of a Poisson process |
| 4.14 | `compound_poisson.py` | Annual aggregate losses |
| 4.15 | `merton_paths.py` | Paths of the Merton jump-diffusion model |
| 4.16 | `merton_price.py` | Merton model: simulation versus the series formula |
| 4.17 | `smile.py` | Implied volatility smiles of the Merton model |
| 4.18 | `jump_returns.py` | Daily returns with and without jumps |
| 4.19 | `corr_paths.py` | Correlated GBM paths and a spread option |

### Chapter 5: Path-Dependent Options
Folder: [`ch05_path_dependent_options/`](ch05_path_dependent_options/)

| Listing | File | Description |
|---|---|---|
| 5.1 | `mcfin_path.py` | Our module for path-dependent options |
| 5.2 | `asian_call.py` | Arithmetic and geometric Asian calls |
| 5.3 | `barrier_paths.py` | Paths of a down-and-out option |
| 5.4 | `barrier_parity.py` | In-out parity for daily monitored barrier calls |
| 5.5 | `barrier_discrete.py` | Discretely monitored down-and-out call |
| 5.6 | `monitoring_bias.py` | Monitoring bias against the number of dates |
| 5.7 | `barrier_bridge.py` | Brownian bridge correction for a continuous barrier |
| 5.8 | `lookback.py` | Lookback call: grid minimum versus bridge minimum |
| 5.9 | `autocallable.py` | Value and characteristics of an autocallable note |
| 5.10 | `model_risk.py` | Heston versus Black-Scholes for path-dependent products |

### Chapter 6: Variance Reduction
Folder: [`ch06_variance_reduction/`](ch06_variance_reduction/)

| Listing | File | Description |
|---|---|---|
| 6.1 | `mcfin_vr.py` | Our module for variance reduction |
| 6.2 | `efficiency.py` | Error and computing time of plain Monte Carlo |
| 6.3 | `antithetic.py` | Antithetic variates for five payoffs |
| 6.4 | `control_stock.py` | The discounted stock as a control variate |
| 6.5 | `control_asian.py` | Control variates for the arithmetic Asian call |
| 6.6 | `crn.py` | Common random numbers for the effect of a barrier shift |
| 6.7 | `stratified.py` | Stratifying the terminal normal for a call option |
| 6.8 | `lhs.py` | Latin hypercube sampling for a three-asset basket option |
| 6.9 | `is_tail.py` | Importance sampling for normal tail probabilities |
| 6.10 | `is_variance.py` | Relative error of importance sampling against the shift |
| 6.11 | `is_call.py` | Importance sampling for out-of-the-money calls |
| 6.12 | `is_credit.py` | Conditional MC and importance sampling for credit |
| 6.13 | `vr_compare.py` | Variance reduction methods for the arithmetic Asian call |

### Chapter 7: Quasi-Monte Carlo Methods
Folder: [`ch07_quasi_monte_carlo_methods/`](ch07_quasi_monte_carlo_methods/)

| Listing | File | Description |
|---|---|---|
| 7.1 | `mcfin_qmc.py` | Our module for quasi-Monte Carlo |
| 7.2 | `vdc.py` | The van der Corput sequence |
| 7.3 | `qmc_points.py` | Pseudo-random, Halton and Sobol points in two dimensions |
| 7.4 | `discrepancy.py` | Centred L^2 discrepancy of random, Halton and Sobol points |
| 7.5 | `qmc_integration.py` | Convergence of MC, Halton and Sobol for a test integral |
| 7.6 | `rqmc_call.py` | Randomised QMC for a European call with error estimates |
| 7.7 | `effective_dim.py` | Variance share of the first normals |
| 7.8 | `asian_qmc.py` | Asian option: MC and Sobol with three constructions |
| 7.9 | `qmc_discontinuous.py` | QMC for smooth and discontinuous payoffs |
| 7.10 | `qmc_basket.py` | Basket option: MC, Latin hypercube and Sobol |

### Chapter 8: Estimating Greeks
Folder: [`ch08_estimating_greeks/`](ch08_estimating_greeks/)

| Listing | File | Description |
|---|---|---|
| 8.1 | `mcfin_greeks.py` | Black-Scholes Greeks and helper functions |
| 8.2 | `fd_delta.py` | Finite-difference deltas with and without common random numbers |
| 8.3 | `fd_variance.py` | Standard error of finite-difference Greeks against the bump size |
| 8.4 | `pathwise.py` | Pathwise delta and vega of a European call |
| 8.5 | `likelihood_ratio.py` | Likelihood ratio Greeks of a European call |
| 8.6 | `gamma_methods.py` | Four estimators of gamma |
| 8.7 | `digital_delta.py` | Delta of a digital call |
| 8.8 | `asian_greeks.py` | Delta and vega of an Asian option by three methods |
| 8.9 | `barrier_delta.py` | Delta of a down-and-out call |
| 8.10 | `delta_hedge.py` | Profit and loss of discrete delta hedging |

### Chapter 9: American Options and the Longstaff--Schwartz Method
Folder: [`ch09_american_options_and_the_longstaff_schwartz_method/`](ch09_american_options_and_the_longstaff_schwartz_method/)

| Listing | File | Description |
|---|---|---|
| 9.1 | `mcfin_amer.py` | Our module for American and Bermudan options |
| 9.2 | `binomial.py` | European and American puts on a binomial tree |
| 9.3 | `lsm_example.py` | The eight-path example of Longstaff and Schwartz |
| 9.4 | `lsm_put.py` | Longstaff-Schwartz price of an American put |
| 9.5 | `lsm_table.py` | LSM against the binomial tree for twelve American puts |
| 9.6 | `lsm_basis.py` | Effect of the basis functions |
| 9.7 | `boundary.py` | Exercise boundaries from LSM and the binomial tree |
| 9.8 | `bermudan.py` | From European to American: more exercise dates |
| 9.9 | `dual_bound.py` | Lower and upper bounds for the Bermudan put |
| 9.10 | `max_call.py` | Bermudan max-call on two assets |

### Chapter 10: Monte Carlo Methods in Risk Management
Folder: [`ch10_monte_carlo_methods_in_risk_management/`](ch10_monte_carlo_methods_in_risk_management/)

| Listing | File | Description |
|---|---|---|
| 10.1 | `mcfin_risk.py` | Our module for risk measurement |
| 10.2 | `var_es_basic.py` | VaR and ES of normal and Student-t losses |
| 10.3 | `var_accuracy.py` | Sampling error of VaR and ES estimates |
| 10.4 | `subadditivity.py` | VaR is not subadditive; ES is |
| 10.5 | `option_book.py` | VaR and ES of an options book |
| 10.6 | `garch_var.py` | Ten-day VaR under GARCH volatility |
| 10.7 | `backtest.py` | Backtesting three VaR models |
| 10.8 | `exposure.py` | Exposure profile and CVA of an FX forward |
| 10.9 | `netting.py` | The effect of netting on exposure and CVA |

## Errata and feedback

Found a mistake or a problem with the code? Please open an
[issue](../../issues) or see [ERRATA.md](ERRATA.md).

## License

The code is released under the MIT License (see [LICENSE](LICENSE)): you may use and adapt it
freely, including in teaching and in your own work. The text of the book is © 2026 Ahmet Göncü,
all rights reserved, and is not part of this repository.
