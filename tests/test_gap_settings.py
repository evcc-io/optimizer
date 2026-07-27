import json
import pathlib

import numpy
import pulp
import pytest

from optimizer import optimizer as opt
from optimizer.optimizer import OBJECTIVE_SCALE, BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

CASE = 'test_cases/012-early-charging-not-perfect.json'


def build(request):
    strategy = request.get('strategy', {})
    grid = request.get('grid', {})
    series = request['time_series']
    return Optimizer(
        strategy=OptimizationStrategy(
            charging_strategy=strategy.get('charging_strategy', 'none'),
            discharging_strategy=strategy.get('discharging_strategy', 'none')),
        grid=GridConfig(p_max_imp=grid.get('p_max_imp'), p_max_exp=grid.get('p_max_exp'),
                        prc_p_exc_imp=grid.get('prc_p_exc_imp')),
        batteries=[BatteryConfig(
            charge_from_grid=bat.get('charge_from_grid', False),
            discharge_to_grid=bat.get('discharge_to_grid', False),
            s_capacity=bat.get('s_capacity', bat['s_max']),
            s_min=bat['s_min'], s_max=bat['s_max'], s_initial=bat['s_initial'],
            p_demand=bat.get('p_demand'), s_goal=bat.get('s_goal'),
            c_min=bat['c_min'], c_max=bat['c_max'], d_max=bat['d_max'], p_a=bat['p_a'],
            c_priority=bat.get('c_priority', 0)) for bat in request['batteries']],
        time_series=TimeSeriesData(dt=series['dt'], gt=series['gt'], ft=series['ft'],
                                   p_N=series['p_N'], p_E=series['p_E']),
        eta_c=request.get('eta_c', 0.95), eta_d=request.get('eta_d', 0.95), M=1e6)


def request():
    return json.loads(pathlib.Path(CASE).read_text())['request']


@pytest.fixture
def captured_solver(monkeypatch):
    """Record what solve() hands to CBC. A gap that never reaches the solver is invisible from
    the outside: the run simply takes longer and still returns a valid schedule, so nothing fails.
    Pin the handover instead of the timing, which would only measure the machine."""
    seen = {}
    original = pulp.PULP_CBC_CMD

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(opt.pulp, 'PULP_CBC_CMD', spy)
    return seen


def test_both_gaps_reach_the_solver(captured_solver):
    model = build(request())
    model.settings.gap_abs = 0.01
    model.settings.gap_rel = 0.002
    model.solve()

    # the absolute gap is in currency units and the model is scaled, so it has to be scaled with it
    assert captured_solver['gapAbs'] == pytest.approx(0.01 * OBJECTIVE_SCALE)
    # the relative gap is a ratio, so it goes over unscaled
    assert captured_solver['gapRel'] == pytest.approx(0.002)


def test_unset_gaps_are_not_passed_as_zero(captured_solver):
    # a zero gap is not the same as no gap: it would demand proven optimality while reading as
    # "unset" in the settings, so both have to arrive as None
    model = build(request())
    model.settings.gap_abs = None
    model.settings.gap_rel = None
    model.solve()

    assert captured_solver['gapAbs'] is None
    assert captured_solver['gapRel'] is None


def test_relative_gap_gives_up_no_more_than_it_promises():
    # the contract of a relative gap: whatever it stops early on is worth less than that share of
    # the objective. Measured on cost_objective, the real economics, against an ungapped solve.
    gap_rel = 0.002

    exact = build(request())
    exact.settings.gap_abs = None
    exact.settings.gap_rel = None
    assert exact.solve()['status'] == 'Optimal'
    optimum = pulp.value(exact.cost_objective)

    gapped = build(request())
    gapped.settings.gap_abs = None
    gapped.settings.gap_rel = gap_rel
    assert gapped.solve()['status'] == 'Optimal'
    stopped = pulp.value(gapped.cost_objective)

    given_up = (optimum - stopped) / abs(optimum)
    assert given_up <= gap_rel, f'gave up {given_up * 100:.4f}%, more than the {gap_rel * 100:g}% gap allows'
    assert numpy.isclose(stopped, optimum, rtol=gap_rel)
