
import numpy

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData


def _optimizer(charging_strategy: str) -> Optimizer:
    # four identical hours, no solar and no household load, so grid import equals battery charging.
    # the charge goal needs less energy than a single time step can deliver, which leaves the
    # optimizer free to choose between one full power step and several partial power steps.
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy=charging_strategy, discharging_strategy='none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(
            charge_from_grid=True,
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
            ft=[0] * 4,
            p_N=[0.3e-3] * 4,
            p_E=[0.0] * 4,
        ),
    )


def test_attenuate_grid_peaks_charges_at_partial_power():
    result = _optimizer('attenuate_grid_peaks').solve()

    assert result['status'] == 'Optimal'

    charging_power = result['batteries'][0]['charging_power']
    # the goal is reached with the charge losses on top, spread evenly over all four time steps
    expected = 4000 / 0.95 / 4

    assert numpy.allclose(charging_power, expected, rtol=1e-3), \
        f"charging power {charging_power}, expected {expected} in every time step"


def test_no_strategy_leaves_charging_unleveled():
    # without the strategy nothing keeps the optimizer from putting all energy into few time steps
    charging_power = _optimizer('none').solve()['batteries'][0]['charging_power']
    expected = 4000 / 0.95 / 4

    assert not numpy.allclose(charging_power, expected, rtol=1e-3)
