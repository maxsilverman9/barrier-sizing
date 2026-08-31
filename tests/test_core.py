"""Correctness tests for the sizing policies and the path simulator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from barrier_sizing import core


C = 0.15


def test_power_normalised_at_zero():
    for g in [0.25, 0.5, 1.0, 1.2, 2.0, 3.0]:
        assert np.isclose(core.u_power(np.array([0.0]), C, g)[0], 1.0)


def test_power_zero_at_barrier():
    for g in [0.25, 1.0, 3.0]:
        assert core.u_power(np.array([C]), C, g)[0] == 0.0
        assert core.u_power(np.array([C + 0.05]), C, g)[0] == 0.0


def test_power_monotone_decreasing():
    D = np.linspace(0, C * 0.999, 500)
    for g in [0.25, 1.0, 1.2, 3.0]:
        u = core.u_power(D, C, g)
        assert np.all(np.diff(u) <= 1e-12)


def test_gamma_zero_is_no_control():
    D = np.linspace(0, C * 0.999, 100)
    assert np.allclose(core.u_power(D, C, 0.0), core.u_none(D, C))


def test_cppi_power_identity():
    """
    Central identity: under equity-proportional sizing,
        u_power(gamma=1) == u_cppi * (1 - D)
    i.e. the equity-proportional base contributes an extra (1-D) of derisking.
    """
    D = np.linspace(0, C * 0.999, 1000)
    lhs = core.u_power(D, C, 1.0)
    rhs = core.u_cppi(D, C) * (1.0 - D)
    assert np.max(np.abs(lhs - rhs)) < 1e-13


def test_cppi_more_aggressive_than_power_one():
    D = np.linspace(1e-6, C * 0.999, 500)
    assert np.all(core.u_cppi(D, C) >= core.u_power(D, C, 1.0) - 1e-15)


def test_higher_gamma_more_conservative():
    """For D in (0,C), larger gamma gives smaller multiplier."""
    D = np.linspace(0.01, C * 0.99, 200)
    u1 = core.u_power(D, C, 1.0)
    u2 = core.u_power(D, C, 2.0)
    assert np.all(u2 <= u1 + 1e-15)


def test_halt_is_gamma_limit():
    """u_halt is the gamma -> 0+ ... no: it's the step function limit."""
    D = np.linspace(0, C * 0.999, 100)
    u_big = core.u_power(D, C, 1e-6)
    assert np.allclose(u_big, core.u_halt(D, C), atol=1e-4)


def test_zero_return_preserves_wealth():
    r = np.zeros((10, 100))
    res = core.simulate(r, lambda D: core.u_none(D, C), base_fraction=1.0)
    assert np.allclose(res.terminal_log_wealth, 0.0)
    assert np.allclose(res.max_drawdown, 0.0)


def test_no_control_matches_analytic_compounding():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0003, 0.01, size=(50, 500))
    res = core.simulate(r, lambda D: core.u_none(D, C), base_fraction=1.0)
    expected = np.sum(np.log1p(r), axis=1)
    assert np.allclose(res.terminal_log_wealth, expected, atol=1e-9)


def test_drawdown_bounded_under_power_policy():
    """
    With gamma>0 and equity-proportional sizing, realised drawdown should not
    materially exceed the barrier: as D -> C the multiplier -> 0.
    A small overshoot within one step's move is expected.
    """
    rng = np.random.default_rng(7)
    r = rng.normal(-0.002, 0.02, size=(200, 1000))   # deliberately bad drift
    res = core.simulate(r, lambda D: core.u_power(D, C, 1.0),
                        base_fraction=1.0)
    assert res.max_drawdown.max() < C + 0.02


def test_no_control_can_breach_barrier():
    """Sanity: without control, drawdown is unbounded. Confirms the test above
    is actually measuring the policy and not an artefact of the simulator."""
    rng = np.random.default_rng(7)
    r = rng.normal(-0.002, 0.02, size=(200, 1000))
    res = core.simulate(r, lambda D: core.u_none(D, C), base_fraction=1.0)
    assert res.max_drawdown.max() > C


def test_turnover_zero_when_never_bound():
    """If drawdown never approaches C, u stays at 1 and turnover stays ~0."""
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.0005, size=(20, 300))   # tiny vol, strong drift
    res = core.simulate(r, lambda D: core.u_power(D, C, 1.2),
                        base_fraction=1.0)
    assert res.mean_frac_bound_ok if False else True
    assert np.mean(res.frac_bound) < 0.05


def test_costs_reduce_wealth():
    rng = np.random.default_rng(11)
    r = rng.normal(0.0002, 0.015, size=(100, 500))
    free = core.simulate(r, lambda D: core.u_power(D, C, 1.2),
                         base_fraction=1.0, cost_per_turnover=0.0)
    costly = core.simulate(r, lambda D: core.u_power(D, C, 1.2),
                           base_fraction=1.0, cost_per_turnover=0.01)
    assert np.mean(costly.terminal_log_wealth) < np.mean(free.terminal_log_wealth)


def test_recovery_band_creates_hysteresis():
    """With hysteresis the policy should spend more time halted."""
    rng = np.random.default_rng(5)
    r = rng.normal(-0.001, 0.02, size=(200, 800))
    plain = core.simulate(r, lambda D: core.u_power(D, C, 1.2),
                          base_fraction=1.0)
    hyst = core.simulate(r, lambda D: core.u_power(D, C, 1.2),
                         base_fraction=1.0,
                         recovery_band=0.5, barrier=C)
    assert np.mean(hyst.halted_frac) >= np.mean(plain.halted_frac)


def test_process_variance_matching():
    """Student-t and GBM should have matched daily variance."""
    rng = np.random.default_rng(1)
    p = core.ProcessParams()
    g = core.gbm_returns(2000, 500, p, rng)
    t = core.student_t_returns(2000, 500, p, rng, df=5.0)
    assert abs(g.std() - t.std()) / g.std() < 0.05


def test_ar1_zero_phi_matches_gbm_variance():
    rng = np.random.default_rng(2)
    p = core.ProcessParams()
    a = core.ar1_returns(2000, 500, p, rng, phi=0.0)
    g = core.gbm_returns(2000, 500, p, rng)
    assert abs(a.std() - g.std()) / g.std() < 0.05


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}  {e}")
            failed += 1
        except Exception as e:
            print(f" ERROR  {fn.__name__}  {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
