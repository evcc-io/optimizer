import json
import pathlib

import numpy
import pulp

from optimizer.optimizer import OBJECTIVE_SCALE, BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

CASE = 'test_cases/012-early-charging-not-perfect.json'


def build(request):
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


def test_objective_split_covers_the_whole_objective():
    # the split must stay exhaustive: cost plus the weighted preferences has to equal what the
    # model maximizes, otherwise the strategy weight would silently drop or double a term
    request = json.loads(pathlib.Path(CASE).read_text())['request']
    optimizer = build(request)
    optimizer.solve()

    combined = (pulp.value(optimizer.cost_objective)
                + optimizer.settings.strategy_weight * pulp.value(optimizer.preference_objective)) * OBJECTIVE_SCALE
    assert numpy.isclose(combined, pulp.value(optimizer.problem.objective), rtol=1e-9), \
        f'cost plus preference {combined}, model objective {pulp.value(optimizer.problem.objective)}'


def test_strategy_weight_does_not_buy_preferences_with_money():
    # the strategies are cost neutral by contract. Weighting them up must not move the economics,
    # which is what breaks first if the weight is raised too far.
    request = json.loads(pathlib.Path(CASE).read_text())['request']

    costs = []
    for weight in (1.0, 3.0):
        optimizer = build(request)
        optimizer.settings.strategy_weight = weight
        assert optimizer.solve()['status'] == 'Optimal'
        costs.append(pulp.value(optimizer.cost_objective))

    plain, weighted = costs
    assert numpy.isclose(weighted, plain, rtol=1e-6), \
        f'weighted cost {weighted}, unweighted cost {plain}'
