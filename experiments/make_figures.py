"""Generate the four figures for the paper."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from barrier_sizing import core

C=0.15
FIG=os.path.join(os.path.dirname(__file__),"..","figures")
plt.rcParams.update({
    "font.family":"serif","font.size":9,"axes.grid":True,
    "grid.alpha":0.25,"grid.linewidth":0.5,"axes.linewidth":0.8,
    "figure.dpi":160,"savefig.dpi":300,"savefig.bbox":"tight",
})

# ---------------------------------------------------------------- Figure 1
def fig1_family():
    D=np.linspace(0,C,400)
    fig,ax=plt.subplots(figsize=(5.0,3.4))
    for g,ls in [(0.5,":"),(1.0,"-"),(1.2,"-"),(2.0,"--"),(5.0,"-.")]:
        lw=1.9 if g in (1.0,1.2) else 1.2
        ax.plot(D/C, core.u_power(D,C,g), ls, lw=lw,
                label=rf"$\gamma={g:g}$"+(" (Grossman–Zhou)" if g==1.0 else ""))
    ax.plot(D/C, core.u_cppi(D,C), color="0.45", lw=1.0,
            label="CPPI")
    ax.set_xlabel(r"drawdown as fraction of barrier,  $D/C$")
    ax.set_ylabel(r"position multiplier  $u(D)$")
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("The power-law sizing family", fontsize=9.5)
    fig.savefig(f"{FIG}/fig1_family.png"); plt.close(fig)
    print("fig1 done")

# ---------------------------------------------------------------- Figure 2
def fig2_boundary():
    import json
    d=json.load(open(f"{os.path.dirname(__file__)}/../data/final_tables.json"))
    t1=[r for r in d["t1"] if r["kmin"] is not None]
    t4=d["t4"]; t3=d["t3"]

    s_sc=[r["sc"] for r in t1]; s_km=[r["kmin"] for r in t1]; s_ks=[r["kstar"] for r in t1]
    r_sc=[r["sc"] for r in t4]; r_km=[r["kmin"] for r in t4]; r_ks=[r["kstar"] for r in t4]
    c_sc=[r["sc"] for r in t3]; c_ks=[r["kstar"] for r in t3]
    c_lbl=[r["asset"] for r in t3]

    fig,ax=plt.subplots(figsize=(5.8,3.9))
    ax.fill_between(r_sc,0,r_km,color="0.88",zorder=0)
    ax.text(3.4,0.78,"halting region",fontsize=7,
            color="0.35",ha="center",va="center")

    ax.plot(s_sc,s_km,"o-",color="0.5",lw=1.1,ms=3.2,
            label=r"$\kappa_{\min}$  simulated (GBM)")
    ax.plot(r_sc,r_km,"o-",color="0.1",lw=1.8,ms=4,
            label=r"$\kappa_{\min}$  realised (SPY momentum)")
    ax.plot(s_sc,s_ks,"s--",color="C0",lw=1.2,ms=3.5,alpha=0.8,
            label=r"$\kappa^*$  simulated (GBM)")
    ax.plot(r_sc,r_ks,"^-",color="C3",lw=1.8,ms=4.5,
            label=r"$\kappa^*$  realised (SPY momentum)")
    ax.plot(c_sc,c_ks,"D",color="C2",ms=4.5,ls="none",
            label=r"$\kappa^*$  cross-asset (native vol)")
    for x,y,l in zip(c_sc,c_ks,c_lbl):
        ax.annotate(l,(x,y),textcoords="offset points",xytext=(4,-8),
                    fontsize=6.5,color="C2")

    ax.set_yscale("log")
    ax.set_yticks([0.6,0.8,1.0,1.5,2,3,5,8])
    ax.set_yticklabels(["0.6","0.8","1.0","1.5","2","3","5","8"])
    ax.annotate("plateau, $\\kappa_{\\min}\\approx1.03$–$1.05$",
                xy=(3.0,1.033), xytext=(1.35,0.63), fontsize=6.8, color="0.1",
                arrowprops=dict(arrowstyle="->",lw=0.7,color="0.1"))
    ax.annotate("growth optimum approaches\na hard stop-out",
                xy=(3.0,7.98), xytext=(1.5,4.6), fontsize=6.8, color="C3",
                arrowprops=dict(arrowstyle="->",lw=0.7,color="C3"))
    ax.set_xlabel(r"strategy volatility relative to barrier,  $\sigma/C$")
    ax.set_ylabel(r"sizing exponent  $\kappa$   (log scale)")
    ax.set_xlim(0.55,4.25)
    ax.legend(frameon=False,fontsize=6.6,loc="lower right")
    ax.set_title(r"Survival boundary is flat; growth optimum diverges",fontsize=9.5)
    fig.savefig(f"{FIG}/fig2_boundary.png"); plt.close(fig)
    print("fig2 done")

# ---------------------------------------------------------------- Figure 3
def fig3_absorbing():
    T=400
    r=np.zeros(T); r[:40]=-0.010; r[40:]=+0.010
    fig,(a1,a2)=plt.subplots(2,1,figsize=(5.2,4.2),sharex=True,
                             gridspec_kw={"height_ratios":[2,1]})
    for g,col,ls in [(0.5,"C3","-"),(1.0,"C0","-")]:
        W=1.0; peak=1.0; Ws=[]; us=[]
        for t in range(T):
            D=max(0.0,(peak-W)/peak)
            u=core.u_power(np.array([D]),C,g)[0]
            Ws.append(W); us.append(u)
            W=W*(1+u*r[t]); peak=max(peak,W)
        a1.plot(Ws,col,ls=ls,lw=1.5,label=rf"$\gamma={g:g}$")
        a2.plot(us,col,ls=ls,lw=1.5)
    bh=np.cumprod(1+r)
    a1.plot(bh,color="0.6",lw=1.0,ls=":",label="uncontrolled")
    a1.axvline(38,color="0.7",lw=0.8,ls="--")
    a1.annotate("barrier reached,\nexposure $\\to$ 0\n(wealth frozen)",
                xy=(45,0.85), xycoords="data",
                xytext=(150,6.5), textcoords="data",
                fontsize=7, color="C3", ha="left",
                arrowprops=dict(arrowstyle="->",lw=0.8,color="C3"))
    a1.set_ylabel("wealth"); a1.legend(frameon=False,fontsize=7.5)
    a1.set_title("The halt state is absorbing",fontsize=9.5)
    a2.set_ylabel(r"$u(D)$"); a2.set_xlabel("time step")
    a2.set_ylim(-0.05,1.05); a2.axvline(38,color="0.7",lw=0.8,ls="--")
    fig.savefig(f"{FIG}/fig3_absorbing.png"); plt.close(fig)
    print("fig3 done")

# ---------------------------------------------------------------- Figure 4
def fig4_realdata():
    px=pd.read_csv(f"{os.path.dirname(__file__)}/../data/prices_daily.csv",
                   index_col=0,parse_dates=True)
    s=px["SPY"].dropna()
    r=s.pct_change().fillna(0.0)
    sig=(s.shift(21)/s.shift(252)-1.0)
    pos=(sig>0).astype(float).shift(1).fillna(0.0)
    strat=(pos*r).values.reshape(1,-1)
    idx=s.index

    fig,(a1,a2)=plt.subplots(2,1,figsize=(5.6,4.4),sharex=True,
                             gridspec_kw={"height_ratios":[2,1]})
    for g,col in [(0.75,"C3"),(1.2,"C0"),(3.0,"C2")]:
        W=1.0; peak=1.0; Ws=[]; DDs=[]
        for t in range(strat.shape[1]):
            D=max(0.0,(peak-W)/peak)
            u=core.u_power(np.array([D]),C,g)[0]
            Ws.append(W); DDs.append(D)
            W=W*(1+u*strat[0,t]); peak=max(peak,W)
        lbl=rf"$\gamma={g:g}$"+(" (halts)" if g==0.75 else "")
        a1.plot(idx,Ws,col,lw=1.3,label=lbl)
        a2.plot(idx,np.array(DDs)*100,col,lw=1.0)
    unc=np.cumprod(1+strat[0])
    a1.plot(idx,unc,color="0.6",lw=1.0,ls=":",label="uncontrolled")
    a1.set_yscale("log"); a1.set_ylabel("wealth (log scale)")
    a1.legend(frameon=False,fontsize=7.5,loc="upper left")
    a1.set_title("SPY 12–1 momentum, 2004–2026",fontsize=9.5)
    a2.axhline(C*100,color="0.4",lw=0.9,ls="--")
    a2.text(idx[60],C*100+0.6,"barrier",fontsize=6.5,color="0.4")
    a2.set_ylabel("drawdown (%)"); a2.set_xlabel("date")
    a2.invert_yaxis()
    fig.savefig(f"{FIG}/fig4_realdata.png"); plt.close(fig)
    print("fig4 done")

if __name__=="__main__":
    fig1_family(); fig2_boundary(); fig3_absorbing(); fig4_realdata()
