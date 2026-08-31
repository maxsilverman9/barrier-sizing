"""
Compute gamma_min (smallest non-halting exponent) on REAL data, to overlay on
Figure 2 alongside the simulated boundary.

gamma_min is the quantity that actually pins gamma* at low sigma/C (Section
5.1). Reporting it for simulation only left the paper's central claim half
tested. This closes that gap.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from barrier_sizing import core

C = 0.15
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "prices_daily.csv")
GRID = np.arange(0.30, 4.01, 0.025)


def momentum(px, lookback=252, skip=21):
    r = px.pct_change().fillna(0.0)
    sig = px.shift(skip) / px.shift(lookback) - 1.0
    pos = (sig > 0).astype(float).shift(1).fillna(0.0)
    return (pos * r).values


def gmin_and_gstar(R):
    """Smallest non-halting exponent, and best admissible exponent."""
    gmin = None
    for g in GRID:
        res = core.simulate(R, lambda D, g=g: core.u_power(D, C, g),
                            base_fraction=1.0)
        if np.mean(res.halted_frac > 0) == 0.0 and \
           res.max_drawdown.max() <= C * 1.10:
            gmin = float(g)
            break
    if gmin is None:
        return None, None
    best = None
    for g in np.arange(gmin, 4.01, 0.05):
        res = core.simulate(R, lambda D, g=g: core.u_power(D, C, g),
                            base_fraction=1.0)
        if np.mean(res.halted_frac > 0) > 0:
            continue
        lw = float(np.mean(res.terminal_log_wealth))
        if best is None or lw > best[1]:
            best = (float(g), lw)
    return gmin, (best[0] if best else None)


def main():
    px = pd.read_csv(DATA, index_col=0, parse_dates=True)
    print("=" * 72)
    print("REAL-DATA HALTING BOUNDARY  (12-1 momentum, 2004-2026)")
    print("=" * 72)

    print("\n--- cross-asset, native volatility ---")
    print(f"  {'asset':>6} {'sigma':>7} {'sigma/C':>8} {'g_min':>7} {'g*':>7}")
    cross = []
    for col in px.columns:
        s = px[col].dropna()
        if len(s) < 600:
            continue
        R = momentum(s).reshape(1, -1)
        vol = float(R.std() * np.sqrt(252))
        gm, gs = gmin_and_gstar(R)
        cross.append((col, vol, vol / C, gm, gs))
        print(f"  {col:>6} {vol:7.1%} {vol/C:8.2f} "
              f"{(f'{gm:.3f}' if gm else 'none'):>7} "
              f"{(f'{gs:.2f}' if gs else 'none'):>7}")

    print("\n--- SPY momentum, volatility-targeted ---")
    print(f"  {'target':>7} {'sigma/C':>8} {'g_min':>7} {'g*':>7}")
    spy = momentum(px["SPY"].dropna())
    base_vol = float(spy.std() * np.sqrt(252))
    vt = []
    for tv in [0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.45, 0.60]:
        R = (spy * (tv / base_vol)).reshape(1, -1)
        gm, gs = gmin_and_gstar(R)
        vt.append((tv, tv / C, gm, gs))
        print(f"  {tv:7.0%} {tv/C:8.2f} "
              f"{(f'{gm:.3f}' if gm else 'none'):>7} "
              f"{(f'{gs:.2f}' if gs else 'none'):>7}")

    out = {
        "cross": [{"asset": a, "sigma": v, "sc": sc, "gmin": gm, "gstar": gs}
                  for a, v, sc, gm, gs in cross],
        "voltarget": [{"target": t, "sc": sc, "gmin": gm, "gstar": gs}
                      for t, sc, gm, gs in vt],
    }
    import json
    with open(os.path.join(os.path.dirname(__file__), "..",
                           "data", "realdata_boundary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> data/realdata_boundary.json")


if __name__ == "__main__":
    main()
