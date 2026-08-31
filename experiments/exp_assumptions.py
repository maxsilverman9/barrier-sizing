"""
Experiment 5.2 -- breaking the Grossman-Zhou assumptions.

The GBM baseline (5.1) reproduces the theoretical prediction gamma* ~= 1 at
moderate leverage. This experiment perturbs one assumption at a time and asks
which direction the optimal exponent moves.

Assumptions under test:
  (a) Gaussian innovations      -> Student-t, variance-matched
  (b) No jumps                  -> Merton jump diffusion, negative jumps
  (c) Frictionless rebalancing  -> proportional cost on |du|
  (d) Serial independence       -> AR(1), momentum and mean reversion

Leverage is held at base_fraction = 1.0 throughout so that any movement in
gamma* is attributable to the assumption break rather than to the sigma/C
ratio effect documented in 5.1.

Pre-registered predictions (recorded before running):
  (a) fat tails      -> gamma* DOWN  (derisk earlier, tails arrive faster)
  (b) jumps          -> gamma* DOWN  (same mechanism, stronger)
  (c) costs          -> gamma* UP    (higher gamma changes size less at small D)
  (d) momentum       -> gamma* UP    (drawdowns persist, hold size longer)
      mean reversion -> gamma* DOWN  (drawdowns revert on their own)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from barrier_sizing import core

C = 0.15
N_PATHS = 4000
N_STEPS = 252 * 5
SEED = 20260826
BASE_F = 1.0
GAMMAS = [0.25, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
ADMISS_TOL = 1.10


def sweep(returns, label, cost=0.0):
    rows = []
    best = None
    for g in GAMMAS:
        res = core.simulate(returns, lambda D, g=g: core.u_power(D, C, g),
                            base_fraction=BASE_F, convention="equity",
                            cost_per_turnover=cost)
        s = core.summarise(res)
        adm = s["worst_max_dd"] <= C * ADMISS_TOL
        rows.append((g, s, adm))
        if adm and (best is None or s["mean_log_wealth"] > best[1]):
            best = (g, s["mean_log_wealth"])

    print(f"\n--- {label} ---")
    print(f"  {'gamma':>6} {'logW':>9} {'p95DD':>7} {'worstDD':>8} "
          f"{'mean_u':>7} {'turn':>7} {'adm':>4}")
    for g, s, adm in rows:
        star = " *" if best and g == best[0] else "  "
        print(f"  {g:6.2f} {s['mean_log_wealth']:+9.4f} {s['p95_max_dd']:7.3f} "
              f"{s['worst_max_dd']:8.3f} {s['mean_u']:7.3f} "
              f"{s['mean_turnover']:7.1f} {('y' if adm else 'N'):>4}{star}")
    if best is None:
        worst = min(r[1]["worst_max_dd"] for r in rows)
        print(f"  => NO ADMISSIBLE GAMMA. Best achievable worst-case DD = "
              f"{worst:.3f} vs barrier {C:.3f}.")
        print(f"     The barrier is not enforceable in this regime.")
        return None
    print(f"  => gamma* = {best[0]:.2f}")
    return best[0]


def run():
    rng = np.random.default_rng(SEED)
    p = core.ProcessParams(mu_annual=0.08, sigma_annual=0.20)

    print("=" * 74)
    print("EXPERIMENT 5.2 -- BREAKING GROSSMAN-ZHOU ASSUMPTIONS")
    print(f"  base_fraction fixed at {BASE_F}; C={C}; "
          f"{N_PATHS} paths x {N_STEPS} steps")
    print("=" * 74)

    results = {}

    R_gbm = core.gbm_returns(N_PATHS, N_STEPS, p, rng)
    results["GBM baseline"] = sweep(R_gbm, "(reference) GBM, frictionless")

    for df in [6.0, 4.0, 3.0]:
        R = core.student_t_returns(N_PATHS, N_STEPS, p, rng, df=df)
        results[f"Student-t df={df:.0f}"] = sweep(
            R, f"(a) fat tails: Student-t, df={df:.0f}")

    for lam, jm in [(2.0, -0.03), (6.0, -0.05)]:
        R = core.jump_diffusion_returns(N_PATHS, N_STEPS, p, rng,
                                        jump_intensity_annual=lam,
                                        jump_mean=jm, jump_std=0.02)
        results[f"jumps lam={lam:.0f}"] = sweep(
            R, f"(b) jump diffusion: {lam:.0f}/yr, mean {jm:+.0%}")

    for c in [0.0005, 0.002, 0.01]:
        results[f"cost={c}"] = sweep(
            R_gbm, f"(c) transaction cost {c*1e4:.0f} bps per unit turnover",
            cost=c)

    for phi in [0.10, 0.05, -0.05, -0.10]:
        R = core.ar1_returns(N_PATHS, N_STEPS, p, rng, phi=phi)
        kind = "momentum" if phi > 0 else "mean reversion"
        results[f"AR1 phi={phi:+.2f}"] = sweep(
            R, f"(d) AR(1) phi={phi:+.2f} ({kind})")

    print("\n" + "=" * 74)
    print("SUMMARY -- movement of gamma* relative to GBM baseline")
    print("=" * 74)
    base = results["GBM baseline"]
    print(f"  {'scenario':<24} {'gamma*':>9} {'shift':>8}")
    for k, v in results.items():
        if v is None:
            print(f"  {k:<24} {'NONE ADM':>9} {'--':>8}   barrier unenforceable")
            continue
        shift = v - base
        arrow = "  ." if abs(shift) < 1e-9 else ("  UP" if shift > 0 else "DOWN")
        print(f"  {k:<24} {v:9.2f} {shift:+8.2f} {arrow}")


if __name__ == "__main__":
    run()
