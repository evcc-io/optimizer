import json
import pathlib

import numpy
import pytest

from optimizer import optimizer as opt
from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

# one case per shape the scaling has to survive: a plain one, one that leans on the peak
# strategies, and one that hits a grid limit and therefore carries penalty terms
CASES = ['012-early-charging-not-perfect',
         '019-unexpected-charge-spikes',
         '021-min-pv-use-case-with-weird-behavior']
STRATEGIES = ['none', 'charge_before_export', 'attenuate_grid_peaks']

# the placement assertion is cheap, it only builds the model, so it runs over every stored case.
# They span three orders in the largest raw coefficient, from 1.6e-1 where market prices go
# negative to 3e2 on the peak levelling ones, the spread a single fixed factor has to cover.
ALL_CASES = sorted(p.stem for p in pathlib.Path('test_cases').glob('*.json'))

# how far below the unscaled result the scaled one may land before it counts as a loss. The two
# are separate CBC runs and CBC converges to about 1e-7 in objective units, so anything tighter
# than that measures the solver's own noise floor instead of the change: on linux the runs sit
# 4e-9 apart on 019 with no strategy. A part per million is still orders below the moves this is
# meant to catch, which are the percent scale losses scaling was introduced to prevent.
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
    """Solve at a fixed scale, or the derived one for None, and return the cost stage's money."""
    original = opt.OBJECTIVE_SCALE
    opt.OBJECTIVE_SCALE = scale
    try:
        model = build(request, charging)
        # solved to proven optimality on both sides. The absolute gap lets either run stop up to a
        # cent short, three orders above the difference this compares, so with it on the assert
        # would measure where the search happened to stop rather than what scaling did.
        model.settings.gap_abs = None
        assert model.solve()['status'] == 'Optimal'
        # the cost stage is what the scaling applies to, read before the preference stage spends
        # its slack. The preference stage derives a factor for its own objective, so a comparison
        # of the total would measure that mechanism instead of this one.
        return model.cost_stage_value
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
    scaled = solve_at(request, charging, None)

    assert scaled >= plain - abs(plain) * TOLERANCE, \
        f'objective at the derived scale: {scaled}, unscaled: {plain}'


@pytest.mark.parametrize('case', ALL_CASES)
@pytest.mark.parametrize('charging', STRATEGIES)
def test_derived_scale_places_the_largest_coefficient(case, charging):
    # what the derived factor is for, and the assertion a fixed one cannot carry: whatever the
    # request's prices and penalty base are, the model the solver receives is handed over in the
    # same numeric window. With a fixed 1e6 the largest coefficient lands anywhere over three
    # orders, up to 3e8 on the peak levelling cases.
    request = json.loads(pathlib.Path(f'test_cases/{case}.json').read_text())['request']
    model = build(request, charging)
    model.create_model()

    largest = max(abs(c) for c in model.problem.objective.values() if c)

    assert numpy.isclose(largest, opt.OBJECTIVE_TARGET, rtol=1e-9), \
        f'largest coefficient {largest:g}, target {opt.OBJECTIVE_TARGET:g}'
