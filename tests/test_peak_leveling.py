import numpy
import pytest

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData


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


def deviation_weight(model, side='imp'):
    """total weight the objective puts on the distance from the level, in the original unit"""
    coefficients = model.problem.objective
    return sum(abs(c) for v, c in coefficients.items()
               if v.name.startswith(f'p_{side}_dev_')) / model.objective_scale


def test_weights_are_ordered_peak_before_deviation():
    # the order the two leveling terms are meant to be read in: lowering the peak beats getting
    # closer to the level, by a decade per W
    model = build()
    assert model.prc_p_peak > model.prc_p_dev


def test_no_ramp_term_is_left_in_the_model():
    # the step to step ramp was dropped: it priced the transitions only, and carrying it alongside
    # the deviation term put two absolute value systems on the same grid power for around a third
    # of the solve time of a leveling case. Guard against it being reintroduced unnoticed.
    model = build(strategy='attenuate_grid_peaks')
    model.create_model()

    assert not hasattr(model, 'prc_p_ramp')
    assert not any(name.startswith('p_imp_ramp') or name.startswith('p_exp_ramp')
                   for name in model.variables)
    assert not any(v.name.startswith(('p_imp_ramp', 'p_exp_ramp'))
                   for v in model.problem.variables())


def test_the_level_goes_blind_on_a_side_resting_at_zero():
    """
    The documented blind spot of the free level, pinned so a change to the term has to confront it.

    On a side that mostly rests at zero the level settles at zero, and with p_grid >= 0 the term
    becomes sum(p_grid[t] * dt[t]) - the energy through the side, which the energy balance already
    fixes. A plateau, a jagged profile and a single spike of the same energy then score the same,
    so nothing below the peak orders them. The ramp term used to; this is what dropping it gives up.
    """
    dt = numpy.array([3600.] * 8)
    plateau = numpy.array([1500., 1500., 1500., 1500., 0., 0., 0., 0.])
    jagged = numpy.array([3000., 0., 3000., 0., 0., 0., 0., 0.])
    spike = numpy.array([6000., 0., 0., 0., 0., 0., 0., 0.])

    def term(power, level=0.):
        """what the objective charges for this profile, the time average of |p - level|"""
        return float((numpy.abs(power - level) * dt).sum() / dt.sum())

    assert term(plateau) == pytest.approx(term(jagged)) == pytest.approx(term(spike))
    # around the mean the three do separate, which is what a mean square deviation would see and
    # what a free level at zero cannot. Kept as the reference the term is measured against.
    assert term(plateau, plateau.mean()) < term(jagged, jagged.mean()) < term(spike, spike.mean())


def test_deviation_weight_does_not_depend_on_the_sampling():
    # the same physical day, once hourly and once in quarter hours. The deviation term is a time
    # average, so both must carry the same total weight - a plain sum over the steps would make the
    # finer sampling weigh four times as much against the peak term, and would price a 15 min
    # excursion like one lasting an hour.
    hourly = build(dt=[3600] * 8)
    quarters = build(dt=[900] * 32)
    hourly.create_model()
    quarters.create_model()

    assert numpy.isclose(deviation_weight(hourly), deviation_weight(quarters), rtol=1e-9)


def test_deviation_variables_exist_only_for_the_leveled_sides():
    demand = build(strategy='attenuate_demand_peaks')
    both = build(strategy='attenuate_grid_peaks')
    none = build(strategy='none')
    for model in (demand, both, none):
        model.create_model()

    assert 'p_imp_dev' in demand.variables and 'p_exp_dev' not in demand.variables
    assert 'p_imp_dev' in both.variables and 'p_exp_dev' in both.variables
    assert 'p_imp_dev' not in none.variables and 'p_exp_dev' not in none.variables


@pytest.mark.parametrize('side, expected', [('imp', 500. + 8000.), ('exp', 300. + 2000.)])
def test_level_is_bounded_by_what_the_side_can_carry(side, expected):
    # import cannot exceed the demand of the worst step plus what the batteries may pull from the
    # grid, export not the production plus what they may push back. Without the bound the new rows
    # relax into big M territory, which costs the solver dearly on the larger models.
    model = build(strategy='attenuate_grid_peaks', ft=[300.] * 8, c_max=8000., d_max=2000.,
                  discharge_to_grid=True)
    model.create_model()

    assert model.variables[f'p_{side}_lvl'].upBound == pytest.approx(expected)
    assert all(v.upBound == pytest.approx(expected) for v in model.variables[f'p_{side}_dev'])


@pytest.mark.parametrize('side, limit', [('imp', {'p_max_exp': 5000.}), ('exp', {'p_max_imp': 5000.})])
def test_a_limit_on_the_other_side_takes_the_bound_back_to_the_big_m(side, limit):
    # the excess variable of a configured limit is not held by the flow direction constraints, so
    # with a limit on the opposite side that side can push this one past what it can carry on its
    # own. Bounding tighter than the big M there would cut off schedules that violate the limit,
    # and those have to stay reachable - they are what the violation flag reports.
    model = build(strategy='attenuate_grid_peaks', ft=[300.] * 8, c_max=8000., d_max=2000.,
                  discharge_to_grid=True, **limit)
    model.create_model()

    assert model.variables[f'p_{side}_lvl'].upBound == model.M


def test_a_pinned_peak_still_levels_the_steps_below_it():
    # a 6 kW load spike the schedule cannot touch fixes the horizon maximum, so the peak term has
    # nothing left to win and the peak term alone would rather charge flat out against the spike
    # and stop than spread the same energy. Regression for the profile 028 stores.
    model = build(gt=[500., 500., 6000., 500., 500., 500., 500., 500.])
    model.batteries[0].s_goal = [0., 0., 0., 0., 0., 20000., 0., 0.]
    result = model.solve()

    assert result['status'] == 'Optimal'
    power = numpy.array(result['grid_import']) * 3600 / numpy.array(model.time_series.dt)
    below = numpy.delete(power, 2)

    assert power.max() == pytest.approx(6000., abs=1.)
    # the five steps carrying the charge sit at one level instead of three of them at the ceiling
    assert below[:5].std() < 1.
