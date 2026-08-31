"""
Power-law drawdown-modulated position sizing: core simulation engine.

Studies the family

    u_gamma(D) = ((C - D) / C) ** gamma        for D < C
               = 0                             for D >= C

where D is the peak-to-current drawdown and C is the barrier.

Benchmarks: no control, fixed-fractional, CPPI, Kelly.

Sizing convention matters and is a first-class parameter here. Two conventions
appear in practice:

  equity-proportional: exposure_t = f * W_t * u(D_t)
  peak-proportional:   exposure_t = f * peak_t * u(D_t)

These are NOT equivalent. In equity-proportional sizing the base already carries
a (1 - D) factor, so the effective control law is strictly more conservative
than the nominal exponent implies. See docs for the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

SizingConvention = Literal["equity", "peak"]


# ──────────────────────────────────────────────────────────────────────
# Position sizing policies
#
# Every policy returns a *multiplier on base exposure*, normalised so that
# u(D=0) == 1.0. This makes all policies directly comparable regardless of
# their internal parameterisation.
# ──────────────────────────────────────────────────────────────────────

def u_power(D: np.ndarray, C: float, gamma: float) -> np.ndarray:
    """Power-law drawdown control. gamma=0 recovers no control."""
    inside = D < C
    out = np.zeros_like(D, dtype=float)
    out[inside] = ((C - D[inside]) / C) ** gamma
    return out


def u_none(D: np.ndarray, C: float) -> np.ndarray:
    """No drawdown control. Always full size."""
    return np.ones_like(D, dtype=float)


def u_cppi(D: np.ndarray, C: float) -> np.ndarray:
    """
    CPPI with floor at peak*(1-C), expressed as a multiplier on
    equity-proportional base exposure and normalised to 1.0 at D=0.

        exposure = m * cushion = m * peak * (C - D)
        u        = exposure / (f * W) = (C - D) / (C * (1 - D))

    Note u_cppi = u_power(gamma=1) / (1 - D): CPPI is strictly more
    aggressive than the gamma=1 power law under equity-proportional sizing.
    """
    inside = D < C
    out = np.zeros_like(D, dtype=float)
    out[inside] = (C - D[inside]) / (C * (1.0 - D[inside]))
    return out


def u_halt(D: np.ndarray, C: float) -> np.ndarray:
    """Binary stop: full size until the barrier, then flat. gamma -> inf limit."""
    return (D < C).astype(float)


# ──────────────────────────────────────────────────────────────────────
# Return processes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ProcessParams:
    mu_annual: float = 0.08
    sigma_annual: float = 0.20
    steps_per_year: int = 252


def gbm_returns(
    n_paths: int, n_steps: int, p: ProcessParams, rng: np.random.Generator
) -> np.ndarray:
    """Geometric Brownian motion. The Grossman-Zhou assumption world."""
    dt = 1.0 / p.steps_per_year
    mu_d = p.mu_annual * dt
    sig_d = p.sigma_annual * np.sqrt(dt)
    return rng.normal(mu_d, sig_d, size=(n_paths, n_steps))


def student_t_returns(
    n_paths: int, n_steps: int, p: ProcessParams, rng: np.random.Generator,
    df: float = 4.0,
) -> np.ndarray:
    """Fat-tailed innovations, variance-matched to the GBM case."""
    dt = 1.0 / p.steps_per_year
    mu_d = p.mu_annual * dt
    sig_d = p.sigma_annual * np.sqrt(dt)
    raw = rng.standard_t(df, size=(n_paths, n_steps))
    raw = raw / np.sqrt(df / (df - 2.0))          # unit variance
    return mu_d + sig_d * raw


def jump_diffusion_returns(
    n_paths: int, n_steps: int, p: ProcessParams, rng: np.random.Generator,
    jump_intensity_annual: float = 2.0,
    jump_mean: float = -0.03,
    jump_std: float = 0.02,
) -> np.ndarray:
    """Merton jump diffusion: diffusive base plus negative Poisson jumps."""
    dt = 1.0 / p.steps_per_year
    base = gbm_returns(n_paths, n_steps, p, rng)
    lam = jump_intensity_annual * dt
    n_jumps = rng.poisson(lam, size=(n_paths, n_steps))
    jump_sizes = rng.normal(jump_mean, jump_std, size=(n_paths, n_steps))
    return base + n_jumps * jump_sizes


def ar1_returns(
    n_paths: int, n_steps: int, p: ProcessParams, rng: np.random.Generator,
    phi: float = 0.05,
) -> np.ndarray:
    """
    AR(1) serial correlation. phi > 0 gives momentum (drawdowns persist),
    phi < 0 gives mean reversion. Unconditional variance is held fixed.
    """
    dt = 1.0 / p.steps_per_year
    mu_d = p.mu_annual * dt
    sig_d = p.sigma_annual * np.sqrt(dt)
    innov_sd = sig_d * np.sqrt(1.0 - phi**2)
    eps = rng.normal(0.0, innov_sd, size=(n_paths, n_steps))
    out = np.empty_like(eps)
    out[:, 0] = rng.normal(0.0, sig_d, size=n_paths)
    for t in range(1, n_steps):
        out[:, t] = phi * out[:, t - 1] + eps[:, t]
    return mu_d + out


# ──────────────────────────────────────────────────────────────────────
# Path simulation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    terminal_log_wealth: np.ndarray   # (n_paths,)
    max_drawdown: np.ndarray          # (n_paths,)
    realised_sharpe: np.ndarray       # (n_paths,) annualised
    turnover: np.ndarray              # (n_paths,) sum |du|
    frac_bound: np.ndarray            # (n_paths,) fraction of steps with u < 1
    recovery_steps: np.ndarray        # (n_paths,) longest peak-to-recovery
    halted_frac: np.ndarray           # (n_paths,) fraction of steps with u == 0
    mean_u: np.ndarray                # (n_paths,) average multiplier
    frac_deep: np.ndarray             # (n_paths,) fraction of steps with u < 0.5


def simulate(
    returns: np.ndarray,
    policy: Callable[[np.ndarray], np.ndarray],
    base_fraction: float,
    steps_per_year: int = 252,
    convention: SizingConvention = "equity",
    cost_per_turnover: float = 0.0,
    recovery_band: float | None = None,
    barrier: float | None = None,
) -> SimResult:
    """
    Evolve wealth under a drawdown-modulated sizing policy.

    The multiplier at step t is computed from the drawdown observed at the end
    of step t-1 and applied to the return realised over step t. No lookahead.

    recovery_band: if set (with barrier), applies hysteresis. Once the policy
    halts (u == 0), it stays halted until D < barrier * recovery_band.
    """
    n_paths, n_steps = returns.shape

    W = np.ones(n_paths)
    peak = np.ones(n_paths)
    max_dd = np.zeros(n_paths)
    u_prev = np.ones(n_paths)
    turnover = np.zeros(n_paths)
    n_bound = np.zeros(n_paths)
    n_halted = np.zeros(n_paths)
    n_deep = np.zeros(n_paths)
    u_sum = np.zeros(n_paths)
    halted = np.zeros(n_paths, dtype=bool)

    # recovery-time bookkeeping
    steps_since_peak = np.zeros(n_paths)
    longest_recovery = np.zeros(n_paths)

    realised = np.empty((n_paths, n_steps))

    for t in range(n_steps):
        D = np.where(peak > 0, (peak - W) / peak, 0.0)
        D = np.maximum(D, 0.0)

        u = policy(D)

        if recovery_band is not None and barrier is not None:
            newly_halted = (u <= 0.0)
            halted = halted | newly_halted
            cleared = halted & (D < barrier * recovery_band)
            halted = halted & ~cleared
            u = np.where(halted, 0.0, u)

        exposure_base = W if convention == "equity" else peak
        port_ret = base_fraction * u * returns[:, t] * (exposure_base / W)

        if cost_per_turnover > 0.0:
            du = np.abs(u - u_prev)
            port_ret = port_ret - cost_per_turnover * base_fraction * du

        realised[:, t] = port_ret
        turnover += np.abs(u - u_prev)
        n_bound += (u < 1.0 - 1e-12)
        n_halted += (u <= 1e-12)
        n_deep += (u < 0.5)
        u_sum += u
        u_prev = u

        W = W * (1.0 + port_ret)
        W = np.maximum(W, 1e-12)          # numerical floor, not a risk control

        new_peak = W >= peak
        longest_recovery = np.maximum(longest_recovery,
                                      np.where(new_peak, steps_since_peak, 0.0))
        steps_since_peak = np.where(new_peak, 0.0, steps_since_peak + 1.0)
        peak = np.maximum(peak, W)
        max_dd = np.maximum(max_dd, (peak - W) / peak)

    longest_recovery = np.maximum(longest_recovery, steps_since_peak)

    mean_r = realised.mean(axis=1)
    std_r = realised.std(axis=1, ddof=1)
    sharpe = np.zeros_like(mean_r)
    nz = std_r > 0
    sharpe[nz] = mean_r[nz] / std_r[nz] * np.sqrt(steps_per_year)

    return SimResult(
        terminal_log_wealth=np.log(W),
        max_drawdown=max_dd,
        realised_sharpe=sharpe,
        turnover=turnover,
        frac_bound=n_bound / n_steps,
        recovery_steps=longest_recovery,
        halted_frac=n_halted / n_steps,
        mean_u=u_sum / n_steps,
        frac_deep=n_deep / n_steps,
    )


def summarise(r: SimResult) -> dict:
    """Cross-path summary statistics."""
    return {
        "mean_log_wealth": float(np.mean(r.terminal_log_wealth)),
        "median_log_wealth": float(np.median(r.terminal_log_wealth)),
        "mean_max_dd": float(np.mean(r.max_drawdown)),
        "p95_max_dd": float(np.percentile(r.max_drawdown, 95)),
        "worst_max_dd": float(np.max(r.max_drawdown)),
        "mean_sharpe": float(np.mean(r.realised_sharpe)),
        "mean_turnover": float(np.mean(r.turnover)),
        "mean_frac_bound": float(np.mean(r.frac_bound)),
        "mean_halted_frac": float(np.mean(r.halted_frac)),
        "mean_u": float(np.mean(r.mean_u)),
        "mean_frac_deep": float(np.mean(r.frac_deep)),
        "mean_recovery_steps": float(np.mean(r.recovery_steps)),
    }
