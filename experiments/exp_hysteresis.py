"""
Experiment 5.3 -- recovery-band hysteresis, and 5.6 -- sizing convention.

5.3: The engine under study halts at D >= C and then refuses to resume until
D < C * rho (rho = 0.5). This is intended to prevent thrashing at the barrier.
Question: does the HALT state ever fire at all under an admissible power-law
policy, and if so does the recovery band help or hurt?

5.6: equity-proportional vs peak-proportional sizing. Section 3 shows these
differ by exactly a factor (1 - D). Does that change gamma*?
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from barrier_sizing import core

C=0.15; N=4000; T=252*5; SEED=20260826
GAMMAS=[0.5,0.75,1.0,1.2,1.5,2.0,3.0]

def main():
    rng=np.random.default_rng(SEED)
    p=core.ProcessParams(0.08,0.20)
    R_gbm=core.gbm_returns(N,T,p,rng)
    R_jump=core.jump_diffusion_returns(N,T,p,rng,jump_intensity_annual=2.0,
                                       jump_mean=-0.03,jump_std=0.02)

    print("="*78)
    print("EXPERIMENT 5.3 -- DOES THE HALT STATE EVER FIRE?")
    print("="*78)
    for name,R,f in [("GBM f=1.0",R_gbm,1.0),
                     ("GBM f=2.0",R_gbm,2.0),
                     ("jumps f=1.0",R_jump,1.0),
                     ("jumps f=2.0",R_jump,2.0)]:
        print(f"\n--- {name} ---")
        print(f"  {'gamma':>6} {'halted%':>9} {'paths_halted':>13} {'worstDD':>8}")
        for g in GAMMAS:
            res=core.simulate(R,lambda D,g=g:core.u_power(D,C,g),
                              base_fraction=f,convention="equity")
            ever = np.mean(res.halted_frac > 0)
            print(f"  {g:6.2f} {np.mean(res.halted_frac)*100:8.4f}% "
                  f"{ever*100:12.2f}% {res.max_drawdown.max():8.3f}")

    print("\n" + "="*78)
    print("EXPERIMENT 5.3b -- RECOVERY BAND EFFECT (where HALT does fire)")
    print("="*78)
    for name,R,f in [("jumps f=2.0",R_jump,2.0)]:
        for g in [1.0,1.5]:
            print(f"\n--- {name}, gamma={g} ---")
            print(f"  {'rho':>6} {'logW':>9} {'worstDD':>8} {'halted%':>9} {'turn':>7}")
            base=core.simulate(R,lambda D,g=g:core.u_power(D,C,g),
                               base_fraction=f,convention="equity")
            sb=core.summarise(base)
            print(f"  {'none':>6} {sb['mean_log_wealth']:+9.4f} "
                  f"{sb['worst_max_dd']:8.3f} {sb['mean_halted_frac']*100:8.3f}% "
                  f"{sb['mean_turnover']:7.1f}")
            for rho in [0.9,0.75,0.5,0.25]:
                res=core.simulate(R,lambda D,g=g:core.u_power(D,C,g),
                                  base_fraction=f,convention="equity",
                                  recovery_band=rho,barrier=C)
                s=core.summarise(res)
                print(f"  {rho:6.2f} {s['mean_log_wealth']:+9.4f} "
                      f"{s['worst_max_dd']:8.3f} {s['mean_halted_frac']*100:8.3f}% "
                      f"{s['mean_turnover']:7.1f}")

    print("\n" + "="*78)
    print("EXPERIMENT 5.6 -- SIZING CONVENTION: equity vs peak proportional")
    print("="*78)
    for f in [1.0,2.0]:
        print(f"\n--- base_fraction={f} ---")
        print(f"  {'gamma':>6} | {'equity logW':>12} {'eq worstDD':>11} | "
              f"{'peak logW':>10} {'pk worstDD':>11}")
        for g in GAMMAS:
            e=core.summarise(core.simulate(R_gbm,lambda D,g=g:core.u_power(D,C,g),
                                base_fraction=f,convention="equity"))
            k=core.summarise(core.simulate(R_gbm,lambda D,g=g:core.u_power(D,C,g),
                                base_fraction=f,convention="peak"))
            print(f"  {g:6.2f} | {e['mean_log_wealth']:+12.4f} {e['worst_max_dd']:11.3f} | "
                  f"{k['mean_log_wealth']:+10.4f} {k['worst_max_dd']:11.3f}")

if __name__=="__main__":
    main()
