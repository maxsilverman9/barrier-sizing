"""
Sensitivity check on the gamma_min ~= 1.05 saturation.

Section 6.1 gives practical guidance resting on the claim that gamma_min
saturates near 1.05 in both simulated and real data. That claim must not be an
artefact of discretisation choices. Three are tested:

  1. grid step        (0.025 -> 0.01, 0.05)
  2. overshoot tol    (1.10 -> 1.00, 1.05, 1.20)
  3. path count       (4000 -> 1000, 8000)  [simulated only]

Additionally: is 1.05 special, or does it drift with the barrier C?
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from barrier_sizing import core

DATA=os.path.join(os.path.dirname(__file__),"..","data","prices_daily.csv")

def momentum(px, lookback=252, skip=21):
    r=px.pct_change().fillna(0.0)
    sig=px.shift(skip)/px.shift(lookback)-1.0
    pos=(sig>0).astype(float).shift(1).fillna(0.0)
    return (pos*r).values

def find_gmin(R, C, step, tol, gmax=6.0):
    for g in np.arange(0.30, gmax+1e-9, step):
        res=core.simulate(R, lambda D,g=g: core.u_power(D,C,g),
                          base_fraction=1.0)
        if np.mean(res.halted_frac>0)==0.0 and res.max_drawdown.max()<=C*tol:
            return float(g)
    return None

def main():
    px=pd.read_csv(DATA,index_col=0,parse_dates=True)
    spy=momentum(px["SPY"].dropna())
    base_vol=float(spy.std()*np.sqrt(252))
    C=0.15

    def R_at(sc):
        tv=sc*C
        return (spy*(tv/base_vol)).reshape(1,-1)

    SCS=[1.00,2.00,3.00,4.00]

    print("="*70)
    print("SENSITIVITY 1 -- GRID STEP  (real data, tol=1.10)")
    print("="*70)
    print(f"  {'sigma/C':>8} {'step=0.05':>10} {'step=0.025':>11} {'step=0.01':>10}")
    for sc in SCS:
        R=R_at(sc)
        vals=[find_gmin(R,C,s,1.10) for s in (0.05,0.025,0.01)]
        print(f"  {sc:8.2f} " + " ".join(
            f"{(f'{v:.3f}' if v else 'none'):>10}" for v in vals))

    print("\n"+"="*70)
    print("SENSITIVITY 2 -- OVERSHOOT TOLERANCE  (real data, step=0.025)")
    print("="*70)
    print(f"  {'sigma/C':>8} {'tol=1.00':>10} {'tol=1.05':>10} {'tol=1.10':>10} {'tol=1.20':>10}")
    for sc in SCS:
        R=R_at(sc)
        vals=[find_gmin(R,C,0.025,t) for t in (1.00,1.05,1.10,1.20)]
        print(f"  {sc:8.2f} " + " ".join(
            f"{(f'{v:.3f}' if v else 'none'):>10}" for v in vals))

    print("\n"+"="*70)
    print("SENSITIVITY 3 -- PATH COUNT  (simulated GBM, sigma/C=2.0)")
    print("="*70)
    p=core.ProcessParams(0.08,0.20)
    for n in (1000,4000,8000):
        rng=np.random.default_rng(20260826)
        R=core.gbm_returns(n,252*5,p,rng)*(2.0*C/0.20)
        g=find_gmin(R,C,0.025,1.10)
        print(f"  n_paths={n:5d}  gamma_min={g:.3f}" if g else
              f"  n_paths={n:5d}  none")

    print("\n"+"="*70)
    print("IS 1.05 SPECIAL?  gamma_min vs barrier C  (real data, sigma/C=3.0)")
    print("="*70)
    print(f"  {'C':>6} {'sigma target':>13} {'gamma_min':>10}")
    for Cv in (0.05,0.10,0.15,0.20,0.30):
        tv=3.0*Cv
        R=(spy*(tv/base_vol)).reshape(1,-1)
        g=find_gmin(R,Cv,0.025,1.10)
        print(f"  {Cv:6.2f} {tv:12.1%} "
              f"{(f'{g:.3f}' if g else 'none'):>10}")

    print("\n"+"="*70)
    print("SATURATION SHAPE -- fine sweep of sigma/C (real data)")
    print("="*70)
    print(f"  {'sigma/C':>8} {'gamma_min':>10}")
    for sc in [0.7,0.9,1.1,1.3,1.6,2.0,2.5,3.0,3.5,4.0,5.0,6.0]:
        g=find_gmin(R_at(sc),C,0.025,1.10)
        print(f"  {sc:8.2f} {(f'{g:.3f}' if g else 'none'):>10}")

if __name__=="__main__":
    main()
