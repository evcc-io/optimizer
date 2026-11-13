import json
import pathlib

import pulp
import pytest

from optimizer.app import app
from optimizer.optimizer import (
    BatteryConfig,
    GridConfig,
    OptimizationStrategy,
    Optimizer,
    TimeSeriesData,
    UncertaintyConfig,
)

# a day in six hourly steps: two cheap night hours, two solar hours, two expensive evening hours.
# The forecast solar fills the battery just far enough to cover the evening, so a shortfall shows
# up as evening import at the high price
PRICES = [1e-4, 1e-4, 2e-4, 2e-4, 4e-4, 4e-4]


def build(uncertainty=None, charge_from_grid=True):
    return Optimizer(
        strategy=OptimizationStrategy('charge_before_export', 'none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=charge_from_grid, discharge_to_grid=False, s_capacity=10000,
                                 s_min=0, s_max=10000, s_initial=2000, c_min=0, c_max=5000, d_max=5000, p_a=1e-4)],
        time_series=TimeSeriesData(dt=[3600] * 6, gt=[1000, 1000, 1000, 1000, 2500, 2500], ft=[0, 0, 3000, 3000, 0, 0],
                                   p_N=PRICES, p_E=[0.5e-4] * 6),
        uncertainty=uncertainty)


def night_import(optimizer):
    return sum(pulp.value(optimizer.variables['n'][t]) for t in range(2))


@pytest.mark.parametrize('uncertainty, ft', [
    (UncertaintyConfig(ft_dev=0.5, gt_dev=0.2, budget=0), None),
    # no solar to fall short and no demand deviation: nothing to hedge
    (UncertaintyConfig(ft_dev=0.5, gt_dev=0, budget=2), [0] * 6),
    # the shortfall stays within the surplus the battery cannot take, which is spilled anyway
    (UncertaintyConfig(ft_dev=0.2, gt_dev=0, budget=2), [0, 0, 20000, 20000, 0, 0]),
])
def test_without_anything_to_hedge_the_model_is_unchanged(uncertainty, ft):
    models = [build(), build(uncertainty)]
    for model in models:
        if ft is not None:
            model.time_series.ft = ft
        model.create_model()
    plain, hedged = (m.problem for m in models)
    assert len(hedged.variables()) == len(plain.variables())
    assert len(hedged.constraints()) == len(plain.constraints())


def test_the_hedge_charges_from_grid_before_an_uncertain_day():
    nominal = build()
    assert nominal.solve()['status'] == 'Optimal'
    # the forecast covers the evening, the night only imports the house load
    assert night_import(nominal) == pytest.approx(2000, abs=1)

    hedged = build(UncertaintyConfig(ft_dev=0.5, budget=2))
    assert hedged.solve()['status'] == 'Optimal'
    assert night_import(hedged) > 2000 + 1000, f'night import {night_import(hedged)}'
    # the hedge costs money on the forecast as sent, which the reported value shows
    assert hedged.get_clean_objective_value() < nominal.get_clean_objective_value()


def test_the_hedge_is_capped_by_its_budget():
    small = build(UncertaintyConfig(ft_dev=0.5, budget=1))
    small.solve()
    large = build(UncertaintyConfig(ft_dev=0.5, budget=2))
    large.solve()
    assert night_import(small) < night_import(large)


def test_a_battery_the_grid_cannot_charge_still_stores_the_solar():
    # stored solar is the reserve that covers a shortfall, so the hedge neither exports it nor holds
    # it back past the evening, where a held reserve only ties the worst case
    nominal = build(charge_from_grid=False)
    nominal.solve()
    hedged = build(UncertaintyConfig(ft_dev=0.5, budget=2), charge_from_grid=False)
    hedged.solve()
    assert sum(pulp.value(hedged.variables['e'][t]) for t in (2, 3)) == pytest.approx(0, abs=1)
    for t in range(6):
        assert pulp.value(hedged.variables['s'][0][t]) == pytest.approx(pulp.value(nominal.variables['s'][0][t]), abs=1)


def test_export_absorbs_the_deviation():
    # twice the solar: the battery is full by the second solar hour and the surplus exported, so a
    # shortfall only reduces export and the schedule stays as it is
    nominal = build()
    nominal.time_series.ft = [0, 0, 8000, 8000, 0, 0]
    nominal.solve()
    hedged = build(UncertaintyConfig(ft_dev=0.3, budget=2))
    hedged.time_series.ft = [0, 0, 8000, 8000, 0, 0]
    assert hedged.solve()['status'] == 'Optimal'
    assert night_import(hedged) == pytest.approx(night_import(nominal), abs=1)


def test_a_flat_tariff_does_not_charge_a_reserve_from_the_grid():
    # a shortfall costs the same whenever it is imported, so a reserve bought from the grid, which
    # only displaces solar into export, cannot beat importing the shortfall when it happens
    hedged = build(UncertaintyConfig(ft_dev=0.3, gt_dev=0.2, budget=2))
    hedged.time_series.p_N = [3e-4] * 6
    assert hedged.solve()['status'] == 'Optimal'
    for t in range(6):
        assert pulp.value(hedged.variables['n'][t]) <= hedged.time_series.gt[t] + 1


@pytest.mark.parametrize('uncertainty', [{'ft_dev': 1.5}, {'gt_dev': -0.1}, {'budget': -1}])
def test_bad_uncertainty_is_rejected(uncertainty):
    request = json.loads(pathlib.Path('test_cases/012-early-charging-not-perfect.json').read_text())['request']
    request['uncertainty'] = uncertainty
    response = app.test_client().post('/optimize/charge-schedule', json=request)
    assert response.status_code == 400
    assert 'uncertainty' in json.dumps(response.json['details'])


def test_a_stored_case_solves_with_a_budget():
    request = json.loads(pathlib.Path('test_cases/012-early-charging-not-perfect.json').read_text())['request']
    request['uncertainty'] = {'ft_dev': 0.3, 'gt_dev': 0.2, 'budget': 8}
    response = app.test_client().post('/optimize/charge-schedule', json=request)
    assert response.status_code == 200
    assert response.json['status'] == 'Optimal'


# three equal days in hourly steps: six cheap night hours, solar around noon, the rest at the high price
def days(uncertainty):
    return Optimizer(
        strategy=OptimizationStrategy('charge_before_export', 'none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=True, discharge_to_grid=False, s_capacity=10000,
                                 s_min=1000, s_max=10000, s_initial=5000, c_min=0, c_max=4000, d_max=5000, p_a=1.8e-4)],
        time_series=TimeSeriesData(dt=[3600] * 72, gt=[500] * 72,
                                   ft=([0] * 8 + [1500, 3000, 4000, 4000, 3000, 1500] + [0] * 10) * 3,
                                   p_N=([2e-4] * 6 + [3e-4] * 18) * 3, p_E=[0.8e-4] * 72),
        eta_c=0.9, eta_d=0.9, uncertainty=uncertainty)


def test_the_hedge_does_not_discharge_into_a_cheap_night_to_charge_it_back():
    # booking the worst case import at the night price must not need a discharge the same night undoes
    hedged = days(UncertaintyConfig(ft_dev=0.6, gt_dev=0.1, budget=6))
    assert hedged.solve()['status'] == 'Optimal'
    for night in (0, 24, 48):
        discharged, charged = (sum(pulp.value(hedged.variables[key][0][t]) for t in range(night, night + 6))
                               for key in ('d', 'c'))
        assert min(discharged, charged) == pytest.approx(0, abs=1)


def test_the_reserve_does_not_grow_with_the_days_planned():
    # the worst case looks back one day, so equal days hold an equal reserve
    hedged = days(UncertaintyConfig(ft_dev=0.4, gt_dev=0.1, budget=6))
    assert hedged.solve()['status'] == 'Optimal'
    lowest = [min(pulp.value(hedged.variables['s'][0][t]) for t in range(day, day + 24)) for day in (24, 48)]
    assert lowest[1] == pytest.approx(lowest[0], abs=1)
