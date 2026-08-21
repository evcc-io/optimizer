import json
import pathlib

import numpy
import pytest
from test_objective_split import build as build_case

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

# the one stored case long enough to make the solve split. The small ones are all proved outright by
# the probe, so none of them ever reaches the tie break's own budget, which is where levelling was
# being lost. 245 steps, 4450 variables, 1263 binaries.
LONG_CASE = '028-attenuate-grid-peaks-long-horizon'


def grid_peaks(request, response):
    """max grid power per side [W]. Kept per side on purpose: attenuate_grid_peaks levels both,
    and a max() across them lets a regression on one hide behind the larger of the two."""
    dt = numpy.array(request['time_series']['dt'], float)
    return {key: float((numpy.array(response[key], float) * 3600 / dt).max())
            for key in ('grid_import', 'grid_export')}


def long_case():
    return json.loads(pathlib.Path(f'test_cases/{LONG_CASE}.json').read_text())


def build(strategy='attenuate_demand_peaks', dt=None, gt=None, ft=None, c_max=8000., d_max=0.,
          charge_from_grid=True, discharge_to_grid=False, p_max_imp=None, p_max_exp=None):
    dt = dt or [3600] * 8
    n = len(dt)
    gt = gt if gt is not None else [500. * dt[t] / 3600. for t in range(n)]
    ft = ft if ft is not None else [0.] * n
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy=strategy, discharging_strategy='none'),
        grid=GridConfig(p_max_imp=p_max_imp, p_max_exp=p_max_exp, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=charge_from_grid, discharge_to_grid=discharge_to_grid,
                                 s_capacity=40000., s_min=0., s_max=40000., s_initial=8000.,
                                 c_min=0., c_max=c_max, d_max=d_max, p_a=0.0001)],
        time_series=TimeSeriesData(dt=dt, gt=gt, ft=ft, p_N=[0.0003] * n, p_E=[0.0001] * n),
        eta_c=0.95, eta_d=0.95, M=1e6)


def import_power(model, result):
    return numpy.array(result['grid_import']) * 3600 / numpy.array(model.time_series.dt)


def test_no_ramp_term_is_left_in_the_model():
    # the step to step ramp was dropped: it priced the transitions only, so it could not tell a
    # straight climb from a plateau, and the second absolute value system over the same grid power
    # cost around a fifth of the solve time of a leveling case. Guard against a silent return.
    model = build(strategy='attenuate_grid_peaks')
    model.create_model()

    assert not hasattr(model, 'prc_p_ramp')
    assert not any(name.startswith(('p_imp_ramp', 'p_exp_ramp')) for name in model.variables)
    assert not any(v.name.startswith(('p_imp_ramp', 'p_exp_ramp')) for v in model.problem.variables())


def test_peak_variables_exist_only_for_the_leveled_sides():
    demand = build(strategy='attenuate_demand_peaks')
    both = build(strategy='attenuate_grid_peaks')
    none = build(strategy='none')
    for model in (demand, both, none):
        model.create_model()

    assert 'p_imp_peak' in demand.variables and 'p_exp_peak' not in demand.variables
    assert 'p_imp_peak' in both.variables and 'p_exp_peak' in both.variables
    assert 'p_imp_peak' not in none.variables and 'p_exp_peak' not in none.variables


def test_the_peak_term_spreads_the_charge_over_the_horizon():
    # what the strategy is for: the same energy taken at partial power over several steps instead
    # of at full power in as few steps as possible
    model = build()
    model.batteries[0].s_goal = [0., 0., 0., 0., 0., 20000., 0., 0.]
    result = model.solve()

    assert result['status'] == 'Optimal'
    power = import_power(model, result)
    # the steps leading up to the goal all carry the same share instead of a few at the ceiling
    assert power[:6].std() < 1.
    assert power.max() < 5000.


def test_a_pinned_peak_leaves_the_steps_below_it_unordered():
    """
    The blind spot of a peak term on its own, pinned so a change to it has to confront the case.

    A 6 kW load spike the schedule cannot touch fixes the horizon maximum. The term prices that one
    value, so once it is pinned nothing is left to win below it: charging flat out against the spike
    and stopping scores exactly as well as spreading the same energy over the window. A step to step
    ramp term used to order those two; this is what dropping it gives up.
    """
    model = build(gt=[500., 500., 6000., 500., 500., 500., 500., 500.])
    model.batteries[0].s_goal = [0., 0., 0., 0., 0., 20000., 0., 0.]
    result = model.solve()

    assert result['status'] == 'Optimal'
    power = import_power(model, result)

    # the spike fixes the maximum and the peak term wins nothing below it
    assert power.max() == pytest.approx(6000., abs=1.)

    # the two readings of the profile below the spike carry the same energy at the same maximum,
    # so the objective scores them alike however differently a mean square deviation would
    flat_out = numpy.array([6000., 6000., 6000., 1044., 1044., 1044., 500., 500.])
    spread = numpy.array([3026., 3026., 6000., 3026., 3026., 3026., 500., 500.])
    dt = numpy.array(model.time_series.dt, float)

    assert (flat_out * dt).sum() == pytest.approx((spread * dt).sum(), rel=1e-3)
    assert flat_out.max() == pytest.approx(spread.max())
    assert spread.std() < flat_out.std() / 1.5


def test_the_long_horizon_case_levels_both_sides():
    # the stored expectation only compares status and objective value, and the tie break is cost
    # neutral by contract, so the objective is identical whether the profile was levelled or not.
    # The peaks are the part worth pinning.
    case = long_case()
    model = build_case(LONG_CASE)
    result = model.solve()

    assert result['status'] == 'Optimal'
    peaks = grid_peaks(case['request'], result)
    stored = grid_peaks(case['request'], case['expected_response'])
    for key, value in peaks.items():
        assert value <= stored[key] + 1, f'{key} peaked at {value} W, stored is {stored[key]} W'


def test_the_long_horizon_case_still_levels_once_the_solve_splits():
    # the regression this guards. The probe cannot prove this request on a production core, so the
    # split runs, and the cost stage used to be handed the whole clock: it is anytime branch and
    # bound and spent all of it, leaving the tie break with 'no time'. The answer came back cost
    # optimal with no levelling at all, 3609 W of export peak against the 1606 W the same money
    # buys, and only a 'Feasible' status to show for it.
    #
    # The threshold sits between the two so the test states which of them came back rather than
    # pinning a schedule: CBC may pick a different optimum of equal value on another build.
    case = long_case()
    model = build_case(LONG_CASE)
    model.settings.probe_seconds = 0      # force the split the probe would otherwise avoid here
    model.settings.time_limit = 5.
    result = model.solve()

    assert result['status'] in ('Optimal', 'Feasible'), result['status']
    peaks = grid_peaks(case['request'], result)
    assert peaks['grid_export'] < 2400, \
        f"export peaked at {peaks['grid_export']} W, unlevelled is 3609 W and levelled 1606 W"

    # the import side is levelled by the MILP tie break, and only reaches 200 W when that stage
    # gets to finish. The pinned LP floor alone reaches 1286 W, which is what a slower machine
    # will see here, so this asserts the floor ran rather than the peak the search would find.
    assert peaks['grid_import'] < 1350, \
        f"import peaked at {peaks['grid_import']} W, unlevelled is 1417 W and the LP floor 1286 W"
