import json
import pathlib

import pulp
import pytest

from optimizer import optimizer as opt
from optimizer.optimizer import OBJECTIVE_SCALE, BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

# one case per shape the scaling has to survive: a plain one, one that leans on the peak
# strategies, and one that hits a grid limit and therefore carries penalty terms
CASES = ['012-early-charging-not-perfect',
         '019-unexpected-charge-spikes',
         '021-min-pv-use-case-with-weird-behavior']
STRATEGIES = ['none', 'charge_before_export', 'attenuate_grid_peaks']

# how far below the unscaled result the scaled one may land before it counts as a loss. The two
# are separate CBC runs and CBC converges to about 1e-7 in objective units, so anything tighter
# than that measures the solver's own noise floor instead of the change: on linux the runs sit
# 4e-9 apart on 019 with no strategy. A part per million is still three orders below the moves
# this is meant to catch, the smallest of which is the 0.4 percent in the strategy weight table.
TOLERANCE = 1e-6


def build(request, charging):
    series = request['time_series']
    grid = request.get('grid', {})
    stored = request.get('strategy', {})
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy=charging,
                                      discharging_strategy=stored.get('discharging_strategy', 'none')),
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


def solve_at(request, charging, scale):
    """Solve at a given OBJECTIVE_SCALE and return the objective back in its original unit."""
    original = opt.OBJECTIVE_SCALE
    opt.OBJECTIVE_SCALE = scale
    try:
        model = build(request, charging)
        # solved to proven optimality on both sides. The absolute gap lets either run stop up to a
        # cent short, three orders above the difference this compares, so with it on the assert
        # would measure where the search happened to stop rather than what scaling did.
        model.settings.gap_abs = None
        assert model.solve()['status'] == 'Optimal'
        return pulp.value(model.problem.objective) / scale
    finally:
        opt.OBJECTIVE_SCALE = original


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('charging', STRATEGIES)
def test_scaling_never_costs_objective_value(case, charging):
    # Scaling is a uniform positive factor, so it cannot move the argmax. What it can move is
    # where the solver stops, and that has to be no worse than without it.
    #
    # Compared on the model objective rather than the reported one on purpose:
    # get_clean_objective_value credits battery gain as s[T-1] - s[0], so two equally optimal
    # solutions that differ in first step charging report different values. Every case here does
    # exactly that once scaled, which would read as a loss of half a percent where the true
    # objective is in fact a shade better.
    request = json.loads(pathlib.Path(f'test_cases/{case}.json').read_text())['request']

    plain = solve_at(request, charging, 1.0)
    scaled = solve_at(request, charging, OBJECTIVE_SCALE)

    assert scaled >= plain - abs(plain) * TOLERANCE, \
        f'objective at scale {OBJECTIVE_SCALE:g}: {scaled}, unscaled: {plain}'
