"""
Experiment 5.1 -- GBM baseline.

Under the Grossman-Zhou assumption set (GBM, log utility, frictionless), the
optimal drawdown-constrained allocation is cushion-proportional. This experiment
sweeps the power-law exponent under exactly those assumptions and asks which
gamma maximises expected terminal log wealth.

This is primarily a CORRECTNESS CHECK on the harness. If the optimum does not
land near the theoretical prediction, something is wrong upstream and no later
result can be trusted.

A second axis matters and is swept jointly: base_fraction. The controller can
only matter if the uncontrolled strategy actually generates drawdowns
comparable to the barrier. Low base_fraction => controller never binds =>
all gammas indistinguishable. This is the mechanism behind the paper's
section 5.5.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from barrier_sizing import core

C = 0.15
N_PATHS = 4000
N_STEPS = 252 * 5          # 5 years daily
SEED = 20260826

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
BASE_FRACTIONS = [0.5, 1.0, 1.5, 2.0, 3.0]


def run():
    rng = np.random.default_rng(SEED)
    p = core.ProcessParams(mu_annual=0.08, sigma_annual=0.20)

    # One shared return tensor so every policy sees identical paths.
    # This removes Monte Carlo noise from the between-policy comparison.
    returns = core.gbm_returns(N_PATHS, N_STEPS, p, rng)

    print("=" * 78)
    print("EXPERIMENT 5.1 -- GBM BASELINE")
    print(f"  paths={N_PATHS}  steps={N_STEPS} ({N_STEPS/252:.0f}y daily)  "
          f"mu={p.mu_annual:.0%}  sigma={p.sigma_annual:.0%}  C={C}")
    print("  sizing convention: equity-proportional")
    print("=" * 78)

    for f in BASE_FRACTIONS:
        # Uncontrolled reference for this leverage level
        ref = core.simulate(returns, lambda D: core.u_none(D, C),
                            base_fraction=f, convention="equity")
        ref_s = core.summarise(ref)

        print(f"\n--- base_fraction = {f:.1f}  "
              f"(portfolio vol ~ {f * p.sigma_annual:.0%}/yr) ---")
        print(f"  uncontrolled: logW={ref_s['mean_log_wealth']:+.4f}  "
              f"meanDD={ref_s['mean_max_dd']:.3f}  "
              f"p95DD={ref_s['p95_max_dd']:.3f}  "
              f"worstDD={ref_s['worst_max_dd']:.3f}")

        print(f"  {'gamma':>6} {'logW':>9} {'medlogW':>9} {'meanDD':>8} "
              f"{'p95DD':>8} {'Sharpe':>8} {'turnover':>9} {'%bound':>8}")

        rows = []
        for g in GAMMAS:
            if g == 0.0:
                pol = lambda D: core.u_none(D, C)
            else:
                pol = lambda D, g=g: core.u_power(D, C, g)
            res = core.simulate(returns, pol, base_fraction=f,
                                convention="equity")
            s = core.summarise(res)
            rows.append((g, s))
            print(f"  {g:6.2f} {s['mean_log_wealth']:+9.4f} "
                  f"{s['median_log_wealth']:+9.4f} "
                  f"{s['mean_max_dd']:8.3f} {s['p95_max_dd']:8.3f} "
                  f"{s['mean_sharpe']:8.3f} {s['mean_turnover']:9.2f} "
                  f"{s['mean_frac_bound']:8.3f}")

        # CPPI comparison
        res_c = core.simulate(returns, lambda D: core.u_cppi(D, C),
                              base_fraction=f, convention="equity")
        s_c = core.summarise(res_c)
        print(f"  {'CPPI':>6} {s_c['mean_log_wealth']:+9.4f} "
              f"{s_c['median_log_wealth']:+9.4f} "
              f"{s_c['mean_max_dd']:8.3f} {s_c['p95_max_dd']:8.3f} "
              f"{s_c['mean_sharpe']:8.3f} {s_c['mean_turnover']:9.2f} "
              f"{s_c['mean_frac_bound']:8.3f}")

        best = max(rows, key=lambda r: r[1]["mean_log_wealth"])
        print(f"  -> argmax_gamma(mean log wealth) = {best[0]:.2f}")


if __name__ == "__main__":
    run()
