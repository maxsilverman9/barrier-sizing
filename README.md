# barrier-sizing

Simulation study of power-law drawdown-modulated position sizing.

This is the reference implementation and full replication package for the paper
*Power-Law Drawdown-Modulated Position Sizing: A Simulation Study of the
Exponent*. Every table and figure in the paper is reproduced by the scripts in
`experiments/`.

## The problem

A drawdown-constrained trading strategy scales its position size down as
drawdown grows. The standard theoretical result (Grossman & Zhou, 1993) says
that under geometric Brownian motion with log utility and frictionless
continuous rebalancing, optimal exposure is proportional to the *cushion*
between wealth and a floor. In practice, implementations use a power law:

```
u(D) = ((C - D) / C) ** kappa       for D < C
u(D) = 0                            for D >= C
```

where `D` is peak-to-current drawdown, `C` is the barrier, and `kappa` is an
exponent usually chosen by convention. `kappa = 1` recovers the
cushion-proportional rule. Larger exponents hold size longer at small
drawdowns and cut harder near the barrier.

This package asks what `kappa` should be, and finds that the question is mostly
answered by a constraint nobody writes down.

## Main findings

**Survival dominates growth.** At low strategy volatility relative to the
barrier, the growth-optimal exponent equals the smallest exponent that avoids
terminating the strategy — the two agree to two decimal places. The objective
contributes nothing; the constraint decides.

**Zero exposure is an absorbing state.** If the multiplier reaches exactly
zero, portfolio returns are zero, so wealth is frozen, so drawdown is frozen,
so the multiplier stays at zero. This holds in a frictionless market with no
distributional assumptions. A corollary: recovery-band hysteresis ("resume when
`D < rho*C`") can never trigger, because the quantity it waits on cannot
change.

**The survival boundary is scale-free.** It depends only on `sigma/C`, and is
invariant to the barrier level across barriers from 5% to 30% — identical to
three decimal places.

**The barrier is not always enforceable.** Under Student-t innovations with
four or fewer degrees of freedom, or jumps arriving six times a year, *no*
exponent satisfies the constraint. A discrete-time multiplicative controller
cannot bound drawdown against gap risk: the jump arrives before the controller
can react.

## Install

```bash
git clone https://github.com/maxsilverman9/barrier-sizing.git
cd barrier-sizing
pip install -r requirements.txt
```

Python 3.10+. Dependencies are numpy, pandas, matplotlib, and (only for
refreshing cached prices) yfinance.

## Quick start

```python
import numpy as np
from barrier_sizing import core

C = 0.15

# The sizing family
D = np.array([0.0, 0.05, 0.10, 0.149])
core.u_power(D, C, kappa=1.2)      # -> [1.0, 0.615, 0.267, 0.001]

# Simulate a strategy under the controller
rng = np.random.default_rng(0)
p = core.ProcessParams(mu_annual=0.08, sigma_annual=0.20)
returns = core.gbm_returns(n_paths=1000, n_steps=1260, p=p, rng=rng)

result = core.simulate(
    returns,
    policy=lambda D: core.u_power(D, C, 1.2),
    base_fraction=1.0,
)
print(core.summarise(result))
```

## Reproducing the paper

Each script prints its results to stdout and is independently runnable.

| Script | Paper section | Runtime |
|---|---|---|
| `experiments/exp_gbm_baseline.py` | 5.1 (initial) | ~1 min |
| `experiments/exp_admissible.py` | 5.1 (corrected admissibility) | ~4 min |
| `experiments/exp_assumptions.py` | 5.2 | ~6 min |
| `experiments/exp_hysteresis.py` | 5.3, 5.6 | ~2 min |
| `experiments/exp_realdata.py` | 5.4 | ~1 min |
| `experiments/exp_realdata_gmin.py` | 5.4 | ~5 min |
| `experiments/exp_sensitivity.py` | 4.3, 5.5 | ~8 min |
| `experiments/exp_final_tables.py` | Tables 1, 3, 4 | ~10 min |
| `experiments/make_figures.py` | Figures 1–4 | ~10 s |

```bash
python experiments/exp_final_tables.py    # final table values
python experiments/make_figures.py        # regenerate figures/
```

All experiments use a fixed seed (`20260826`) and are deterministic.

## A note on the admissibility criterion

This is the methodological point most likely to matter to someone reusing this
code.

**Realised maximum drawdown is not a sufficient test of whether a policy
satisfied its constraint.** A policy that reaches the barrier and freezes
reports `max_dd ≈ C`, which is indistinguishable from a policy that controlled
risk successfully. An earlier version of this study used that criterion alone
and reached materially different conclusions — low exponents appeared
admissible when they had in fact terminated.

The criterion used here requires both:

1. the policy never halts: `P(exists t : u(D_t) == 0) == 0`, and
2. `max_dd <= 1.10 * C`, allowing for single-period overshoot.

Condition (1) is the one that matters. Any study using drawdown alone as its
constraint check has the same bug available to it.

## Data

`data/prices_daily.csv` contains daily adjusted closes for SPY, QQQ, IWM, TLT
and GLD, January 2004 to July 2026, retrieved via yfinance. It is cached so
the experiments are reproducible without network access. To refresh it, see the
header of `experiments/exp_realdata.py`.

## Tests

```bash
python tests/test_core.py
```

17 tests covering the policy identities (including the CPPI relationship to
1e-13), monotonicity, the absorbing-state proposition on a constructed path,
agreement between the uncontrolled simulator and analytic compounding, and
variance matching across return processes.

## Citation

```
@misc{silverman2026barrier,
  author = {Silverman, Max},
  title  = {Power-Law Drawdown-Modulated Position Sizing:
            A Simulation Study of the Exponent},
  year   = {2026},
  note   = {Working paper}
}
```

## License

MIT. See `LICENSE`.
