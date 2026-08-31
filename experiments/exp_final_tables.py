"""
Final table values: extended exponent grid (to 12.0) so kappa* is not
grid-limited at high sigma/C, plus fine-grid cross-asset recomputation.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from barrier_sizing import core

C=0.15
DATA=os.path.join(os.path.dirname(__file__),"..","data","prices_daily.csv")

def momentum(px, lookback=252, skip=21):
    r=px.pct_change().fillna(0.0)
    sig=px.shift(skip)/px.shift(lookback)-1.0
    pos=(sig>0).astype(float).shift(1).fillna(0.0)
    return (pos*r).values

def _ok(R,C,k):
    res=core.simulate(R, lambda D,k=k: core.u_power(D,C,k), base_fraction=1.0)
    return (np.mean(res.halted_frac>0)==0.0 and
            res.max_drawdown.max()<=C*1.10,
            float(np.mean(res.terminal_log_wealth)))

def boundary(R, C):
    # bisect for kmin on [0.30, 12.0]; admissibility is monotone in k
    lo,hi=0.30,12.0
    if not _ok(R,C,hi)[0]: return None,None,None
    if _ok(R,C,lo)[0]: kmin=lo
    else:
        for _ in range(14):
            mid=0.5*(lo+hi)
            if _ok(R,C,mid)[0]: hi=mid
            else: lo=mid
        kmin=hi
    kmin=round(kmin,3)
    # coarse then refine for kstar
    grid=np.arange(kmin,12.01,0.25)
    vals=[(k,_ok(R,C,k)) for k in grid]
    vals=[(k,v[1]) for k,v in vals if v[0]]
    if not vals: return kmin,None,None
    kb=max(vals,key=lambda t:t[1])[0]
    fine=np.arange(max(kmin,kb-0.25),kb+0.26,0.05)
    vals2=[(k,_ok(R,C,k)) for k in fine]
    vals2=[(k,v[1]) for k,v in vals2 if v[0]]
    kbest,lw=max(vals2,key=lambda t:t[1])
    return kmin,float(kbest),float(lw)

px=pd.read_csv(DATA,index_col=0,parse_dates=True)

print("="*78)
print("TABLE 3 (final) -- cross-asset, native volatility, grid to 12.0")
print("="*78)
print(f"  {'asset':>6} {'sigma':>7} {'s/C':>6} {'unc maxDD':>10} {'k_min':>7} {'k*':>7} {'logW':>9}")
t3=[]
for col in px.columns:
    s=px[col].dropna()
    if len(s)<600: continue
    R=momentum(s).reshape(1,-1)
    vol=float(R.std()*np.sqrt(252))
    unc=core.summarise(core.simulate(R,lambda D:core.u_none(D,C),base_fraction=1.0))
    kmin,ks,lw=boundary(R,C)
    t3.append(dict(asset=col,sigma=vol,sc=vol/C,unc_dd=unc["worst_max_dd"],
                   kmin=kmin,kstar=ks,logw=lw))
    print(f"  {col:>6} {vol:7.1%} {vol/C:6.2f} {unc['worst_max_dd']:10.3f} "
          f"{kmin:7.3f} {ks:7.2f} {lw:+9.4f}")

print("\n"+"="*78)
print("TABLE 4 (final) -- SPY vol-targeted, grid to 12.0")
print("="*78)
spy=momentum(px["SPY"].dropna()); bv=float(spy.std()*np.sqrt(252))
print(f"  {'target':>7} {'s/C':>6} {'k_min':>7} {'k*':>7} {'logW':>9}")
t4=[]
for tv in [0.10,0.15,0.20,0.30,0.45,0.60]:
    R=(spy*(tv/bv)).reshape(1,-1)
    kmin,ks,lw=boundary(R,C)
    t4.append(dict(target=tv,sc=tv/C,kmin=kmin,kstar=ks,logw=lw))
    print(f"  {tv:7.0%} {tv/C:6.2f} {kmin:7.3f} {ks:7.2f} {lw:+9.4f}")

print("\n"+"="*78)
print("TABLE 1 (final) -- simulated GBM, grid to 12.0")
print("="*78)
rng=np.random.default_rng(20260826)
p=core.ProcessParams(0.08,0.20)
Rg=core.gbm_returns(4000,252*5,p,rng)
print(f"  {'f':>5} {'s/C':>6} {'k_min':>7} {'k*':>7}")
t1=[]
for f in [0.5,0.75,1.0,1.5,2.0,3.0]:
    Rs=Rg*f
    kmin,ks,lw=boundary(Rs,C)
    t1.append(dict(f=f,sc=f*0.20/C,kmin=kmin,kstar=ks))
    print(f"  {f:5.2f} {f*0.20/C:6.2f} "
          f"{(f'{kmin:.3f}' if kmin else 'none'):>7} "
          f"{(f'{ks:.2f}' if ks else 'none'):>7}")

json.dump(dict(t1=t1,t3=t3,t4=t4),
          open(os.path.join(os.path.dirname(__file__),"..","data",
                            "final_tables.json"),"w"),indent=2)
print("\nsaved -> data/final_tables.json")
