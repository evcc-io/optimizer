
import numpy
import pytest

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

# charge energy needed to reach the 4 kWh goal, including the charge losses
EXPECTED_CHARGE = 4000 / 0.95

# charging spread evenly over all four steps, which is the flat grid import profile
LEVELED_DEMAND = [EXPECTED_CHARGE / 4] * 4
# charging placed on the two high yield steps, which is the flat grid export profile
SHAVED_FEEDIN = [0, EXPECTED_CHARGE / 2, EXPECTED_CHARGE / 2, 0]


def _optimizer(charging_strategy: str, ft: list, charge_from_grid: bool, p_E: float) -> Optimizer:
    # four identical hours with a flat price profile, so placing the charge energy is cost neutral.
    # the goal needs less energy than a single time step can deliver, which leaves the optimizer
    # free to choose between one full power step and several partial power steps.
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy=charging_strategy, discharging_strategy='none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(
            charge_from_grid=charge_from_grid,
            discharge_to_grid=False,
            s_capacity=20000,
            s_min=0,
            s_max=20000,
            s_initial=0,
            c_min=0,
            c_max=8000,
            d_max=0,
            p_a=0,
            s_goal=[0, 0, 0, 4000],
        )],
        time_series=TimeSeriesData(
            dt=[3600] * 4,
            gt=[0] * 4,
            ft=ft,
            p_N=[0.3e-3] * 4,
            p_E=[p_E] * 4,
        ),
    )


def _demand_case(charging_strategy: str) -> Optimizer:
    # no solar and no household load, so grid import equals battery charging
    return _optimizer(charging_strategy, ft=[0] * 4, charge_from_grid=True, p_E=0.0)


def _feedin_case(charging_strategy: str) -> Optimizer:
    # solar surplus with a hump in the middle and no grid charging, so the battery can only take
    # energy out of the export. the export peak drops only if charging happens inside the hump.
    return _optimizer(charging_strategy, ft=[2000, 6000, 6000, 2000], charge_from_grid=False, p_E=0.1e-3)


@pytest.mark.parametrize('charging_strategy', ['attenuate_demand_peaks', 'attenuate_grid_peaks'])
def test_demand_peaks_are_leveled(charging_strategy):
    result = _demand_case(charging_strategy).solve()

    assert result['status'] == 'Optimal'
    charging_power = result['batteries'][0]['charging_power']

    assert numpy.allclose(charging_power, LEVELED_DEMAND, rtol=1e-3), \
        f"charging power {charging_power}, expected {LEVELED_DEMAND}"


@pytest.mark.parametrize('charging_strategy', ['attenuate_feedin_peaks', 'attenuate_grid_peaks'])
def test_feedin_peaks_are_shaved(charging_strategy):
    result = _feedin_case(charging_strategy).solve()

    assert result['status'] == 'Optimal'
    charging_power = result['batteries'][0]['charging_power']

    assert numpy.allclose(charging_power, SHAVED_FEEDIN, atol=1), \
        f"charging power {charging_power}, expected {SHAVED_FEEDIN}"


def test_feedin_strategy_leaves_demand_peaks_unleveled():
    # the feed-in option must not level grid import, otherwise the options are not separable
    charging_power = _demand_case('attenuate_feedin_peaks').solve()['batteries'][0]['charging_power']

    assert not numpy.allclose(charging_power, LEVELED_DEMAND, rtol=1e-3)


def test_demand_strategy_leaves_feedin_peaks_unshaved():
    # the demand option must not shave grid export, otherwise the options are not separable
    charging_power = _feedin_case('attenuate_demand_peaks').solve()['batteries'][0]['charging_power']

    assert not numpy.allclose(charging_power, SHAVED_FEEDIN, atol=1)


def test_no_strategy_leaves_charging_unleveled():
    # without a strategy nothing keeps the optimizer from putting all energy into few time steps
    assert not numpy.allclose(
        _demand_case('none').solve()['batteries'][0]['charging_power'], LEVELED_DEMAND, rtol=1e-3)
    assert not numpy.allclose(
        _feedin_case('none').solve()['batteries'][0]['charging_power'], SHAVED_FEEDIN, atol=1)
