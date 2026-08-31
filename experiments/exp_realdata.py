"""
Experiment 5.4 -- real market data.

Applies the power-law family to a simple, fully-specified momentum strategy on
liquid ETFs, 2004-2026. The period deliberately spans 2008, 2020 and 2022 so
the controller is exercised in regimes where drawdowns approach the barrier.

The underlying strategy is intentionally unremarkable: 12-1 time-series
momentum, long/flat, monthly rebalanced. The paper is about the sizing layer,
not the signal. Using a well-known signal keeps the result attributable to the
controller.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from barrier_sizing import core

C=0.15
GAMMAS=[0.5,0.75,0.9,1.0,1.1,1.2,1.35,1.5,2.0,3.0]
DATA=os.path.join(os.path.dirname(__file__),"..","data","prices_daily.csv")

def momentum_returns(px: pd.Series, lookback=252, skip=21):
    """12-1 momentum: long if trailing (12m ex last 1m) return > 0, else flat."""
    r = px.pct_change().fillna(0.0)
    sig_src = px.shift(skip) / px.shift(lookback) - 1.0
    pos = (sig_src > 0).astype(float).shift(1).fillna(0.0)   # no lookahead
    return (pos * r).values, r.values

def run_one(strat_r, label, target_vol=None):
    strat_r = np.asarray(strat_r, dtype=float).reshape(1,-1)
    n = strat_r.shape[1]
    ann_vol = strat_r.std()*np.sqrt(252)
    scale = 1.0 if target_vol is None else (target_vol/ann_vol if ann_vol>0 else 1.0)
    R = strat_r*scale
    eff_vol = R.std()*np.sqrt(252)

    print(f"\n--- {label}  (n={n}, realised vol {eff_vol:.1%}/yr, "
          f"sigma/C={eff_vol/C:.2f}) ---")
    ref=core.summarise(core.simulate(R,lambda D:core.u_none(D,C),base_fraction=1.0))
    print(f"  {'policy':>8} {'logW':>9} {'maxDD':>7} {'Sharpe':>7} "
          f"{'halted%':>8} {'mean_u':>7} {'verdict':>9}")
    print(f"  {'NONE':>8} {ref['mean_log_wealth']:+9.4f} {ref['worst_max_dd']:7.3f} "
          f"{ref['mean_sharpe']:7.3f} {0.0:7.2f}% {1.0:7.3f} "
          f"{('breach' if ref['worst_max_dd']>C else 'ok'):>9}")
    best=None
    for g in GAMMAS:
        res=core.simulate(R,lambda D,g=g:core.u_power(D,C,g),base_fraction=1.0)
        s=core.summarise(res); halted=float(np.mean(res.halted_frac>0))
        adm = (halted==0.0) and s["worst_max_dd"]<=C*1.10
        if adm and (best is None or s["mean_log_wealth"]>best[1]):
            best=(g,s["mean_log_wealth"])
        v = "ok" if adm else ("HALTS" if halted>0 else "breach")
        print(f"  {('g='+format(g,'.2f')):>8} {s['mean_log_wealth']:+9.4f} "
              f"{s['worst_max_dd']:7.3f} {s['mean_sharpe']:7.3f} "
              f"{halted*100:7.2f}% {s['mean_u']:7.3f} {v:>9}")
    print(f"  => gamma* = {best[0]:.2f}" if best else "  => NO ADMISSIBLE GAMMA")
    return best

def main():
    px = pd.read_csv(DATA, index_col=0, parse_dates=True)
    print("="*80)
    print("EXPERIMENT 5.4 -- REAL DATA: 12-1 MOMENTUM, DAILY, 2004-2026")
    print(f"  assets: {list(px.columns)}   {px.index[0].date()} -> {px.index[-1].date()}")
    print("="*80)

    for col in px.columns:
        s = px[col].dropna()
        if len(s) < 600: continue
        strat, _ = momentum_returns(s)
        run_one(strat, f"{col} momentum (native sizing)")

    # Volatility-targeted versions so sigma/C spans the range studied in 5.1R
    print("\n" + "="*80)
    print("VOL-TARGETED: sweeping sigma/C to test the 5.1R boundary prediction")
    print("  5.1R predicts gamma_min rises with sigma/C and the barrier becomes")
    print("  unenforceable above sigma/C ~ 3.")
    print("="*80)
    spy = px["SPY"].dropna()
    strat,_ = momentum_returns(spy)
    for tv in [0.10,0.15,0.20,0.30,0.45,0.60]:
        run_one(strat, f"SPY momentum @ {tv:.0%} vol target", target_vol=tv)

if __name__=="__main__":
    main()
