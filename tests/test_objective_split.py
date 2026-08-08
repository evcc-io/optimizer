import json
import pathlib

import numpy
import pulp
import pytest

from optimizer.optimizer import COST_BOUND_SLACK, BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

# a plain case, one that leans on the priorities, and one that levels grid peaks. The last is the
# one where the preferences are worth enough real money to be tempted to buy some
CASES = ['012-early-charging-not-perfect',
         '017-battery-charge-priotization-2',
         '026-attenuate-grid-peaks']


def build(case):
    request = json.loads(pathlib.Path(f'test_cases/{case}.json').read_text())['request']
    strategy_data = request.get('strategy', {})
    grid_data = request.get('grid', {})
    series = request['time_series']
    return Optimizer(
        strategy=OptimizationStrategy(
            charging_strategy=strategy_data.get('charging_strategy', 'none'),
            discharging_strategy=strategy_data.get('discharging_strategy', 'none')),
        grid=GridConfig(p_max_imp=grid_data.get('p_max_imp'), p_max_exp=grid_data.get('p_max_exp'),
                        prc_p_exc_imp=grid_data.get('prc_p_exc_imp')),
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


def solve_cost_only(optimizer):
    """First stage on its own, solved to proven optimality."""
    optimizer.create_model()
    optimizer.problem.setObjective(optimizer.cost_objective * optimizer.objective_scale)
    optimizer.problem.solve(pulp.PULP_CBC_CMD(msg=0))
    return optimizer


@pytest.mark.parametrize('case', CASES)
def test_objective_split_covers_the_whole_objective(case):
    # the split must stay exhaustive: cost plus preferences has to equal what a single solve would
    # maximize, otherwise a term would silently belong to neither stage
    optimizer = build(case)
    optimizer.solve()

    combined = (pulp.value(optimizer.cost_objective)
                + pulp.value(optimizer.preference_objective)) * optimizer.objective_scale
    assert numpy.isclose(combined, pulp.value(optimizer.problem.objective), rtol=1e-9), \
        f'cost plus preference {combined}, model objective {pulp.value(optimizer.problem.objective)}'


@pytest.mark.parametrize('case', CASES)
def test_preferences_are_not_paid_for_with_money(case):
    # the strategies are cost neutral by contract. The second stage may reorder a schedule but it
    # may not buy a better tie break with real money, which is what the cost bound is there for.
    # The probe is disabled so the split actually runs: these cases are small enough that the
    # joint solve proves them outright, which is the point of the probe but not what this tests.
    optimizer = build(case)
    optimizer.settings.probe_seconds = 0
    assert optimizer.solve()['status'] == 'Optimal'
    assert optimizer.solve_path == 'split', f'took the {optimizer.solve_path} path'
    assert optimizer.preference_stage == 'Optimal', f'preference stage ended as {optimizer.preference_stage}'

    # measured against what the cost stage had before the tie break ran, so the gap the cost stage
    # is allowed to stop on does not enter. The slack is what _solve_preferences hands over, plus
    # CBC's own feasibility tolerance, which it may violate that bound row by.
    allowed = optimizer.settings.preference_budget + COST_BOUND_SLACK + 1e-6
    assert pulp.value(optimizer.cost_objective) >= optimizer.cost_stage_value - allowed, \
        f'cost after the tie break {pulp.value(optimizer.cost_objective)}, before {optimizer.cost_stage_value}'


@pytest.mark.parametrize('case', CASES)
def test_preference_stage_decides_the_tie(case):
    # what the second stage is for: the first stage is indifferent between the schedules it leaves
    # equally priced, so its preference value is whatever the search happened to stop on. Deciding
    # the tie afterwards has to be at least as good, and on these cases it is strictly better.
    undecided = pulp.value(solve_cost_only(build(case)).preference_objective)

    decided = build(case)
    decided.settings.probe_seconds = 0
    assert decided.solve()['status'] == 'Optimal'

    assert pulp.value(decided.preference_objective) > undecided, \
        f'preferences after the tie break {pulp.value(decided.preference_objective)}, before {undecided}'


@pytest.mark.parametrize('case', CASES)
def test_easy_requests_never_reach_the_split(case):
    # the whole point of the probe: a model that proves optimality on the joint objective decided
    # its own tie, in one solve, with no gap and no preference budget to give anything away. Every
    # stored case is that shape, so none of them should be paying for a second solve.
    optimizer = build(case)
    assert optimizer.solve()['status'] == 'Optimal'
    assert optimizer.solve_path == 'joint', f'took the {optimizer.solve_path} path'

    # and it is the better answer. Compared on cost plus preference, which is what the model
    # maximizes: the split can edge ahead on the preferences alone because its cost bound lets it
    # spend COST_BOUND_SLACK buying them, so preferences on their own would score the wrong thing.
    split = build(case)
    split.settings.probe_seconds = 0
    split.solve()
    joint_total = pulp.value(optimizer.cost_objective) + pulp.value(optimizer.preference_objective)
    split_total = pulp.value(split.cost_objective) + pulp.value(split.preference_objective)
    assert joint_total >= split_total - 1e-9, f'joint {joint_total}, split {split_total}'
