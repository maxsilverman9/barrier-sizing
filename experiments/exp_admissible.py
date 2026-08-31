"""
Experiment 5.1R -- corrected admissibility.

5.3 established that the halt state (u=0) is ABSORBING: wealth freezes, so
drawdown freezes, so the multiplier never recovers. A policy that halts has
not satisfied the drawdown constraint -- it has terminated.

The original admissibility test in 5.1 used only worst-case realised drawdown,
which cannot distinguish "controlled the drawdown" from "hit the barrier and
died". This re-runs the sweep with halting as a disqualifier.

Admissible <=> P(ever halt) == 0 AND worst realised DD <= C * tol.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from barrier_sizing import core

C=0.15; N=4000; T=252*5; SEED=20260826
GAMMAS=[0.25,0.5,0.75,0.9,1.0,1.1,1.2,1.35,1.5,2.0,3.0]
TOL=1.10

def main():
    rng=np.random.default_rng(SEED)
    p=core.ProcessParams(0.08,0.20)
    R=core.gbm_returns(N,T,p,rng)

    print("="*86)
    print("EXPERIMENT 5.1R -- GAMMA SWEEP WITH HALTING AS DISQUALIFIER")
    print("  A policy that halts has terminated, not survived. See 5.3.")
    print("="*86)

    for f in [0.5,1.0,1.5,2.0,3.0]:
        print(f"\n--- base_fraction={f:.1f} (vol ~{f*0.20:.0%}/yr, sigma/C={f*0.20/C:.2f}) ---")
        print(f"  {'gamma':>6} {'logW':>9} {'worstDD':>8} {'P(halt)':>9} "
              f"{'halt%steps':>11} {'mean_u':>7} {'verdict':>10}")
        best=None
        for g in GAMMAS:
            res=core.simulate(R,lambda D,g=g:core.u_power(D,C,g),
                              base_fraction=f,convention="equity")
            s=core.summarise(res)
            p_halt=float(np.mean(res.halted_frac>0))
            dd_ok = s["worst_max_dd"]<=C*TOL
            adm = (p_halt==0.0) and dd_ok
            if adm and (best is None or s["mean_log_wealth"]>best[1]):
                best=(g,s["mean_log_wealth"])
            verdict = "ok" if adm else ("HALTS" if p_halt>0 else "breach")
            star=" *" if best and g==best[0] else "  "
            print(f"  {g:6.2f} {s['mean_log_wealth']:+9.4f} {s['worst_max_dd']:8.3f} "
                  f"{p_halt*100:8.2f}% {s['mean_halted_frac']*100:10.3f}% "
                  f"{s['mean_u']:7.3f} {verdict:>10}{star}")
        if best:
            # locate the halting boundary
            print(f"  => gamma* = {best[0]:.2f}   (logW {best[1]:+.4f})")
        else:
            print(f"  => NO ADMISSIBLE GAMMA in tested range")

    # Where exactly is the halting boundary as a function of sigma/C?
    print("\n" + "="*86)
    print("HALTING BOUNDARY: minimum gamma with P(halt)=0")
    print("="*86)
    print(f"  {'f':>5} {'sigma/C':>8} {'gamma_min':>10} {'gamma*':>8}")
    fine=np.arange(0.30,2.51,0.05)
    for f in [0.5,0.75,1.0,1.5,2.0,3.0,4.0]:
        gmin=None
        for g in fine:
            res=core.simulate(R,lambda D,g=g:core.u_power(D,C,g),
                              base_fraction=f,convention="equity")
            if np.mean(res.halted_frac>0)==0.0:
                gmin=g; break
        # gamma* among admissible
        bg=None
        if gmin is not None:
            for g in np.arange(gmin,3.01,0.05):
                res=core.simulate(R,lambda D,g=g:core.u_power(D,C,g),
                                  base_fraction=f,convention="equity")
                if np.mean(res.halted_frac>0)>0: continue
                lw=float(np.mean(res.terminal_log_wealth))
                if bg is None or lw>bg[1]: bg=(g,lw)
        print(f"  {f:5.2f} {f*0.20/C:8.2f} "
              f"{(f'{gmin:.2f}' if gmin else 'none'):>10} "
              f"{(f'{bg[0]:.2f}' if bg else 'none'):>8}")

if __name__=="__main__":
    main()
