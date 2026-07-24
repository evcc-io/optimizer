import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from optimizer.optimizer import (  # noqa: E402
    BatteryConfig,
    GridConfig,
    OptimizationStrategy,
    Optimizer,
    TimeSeriesData,
)


def _optimize(charging_strategy, ft, gt, c_max, s_max):
    """single battery, 1 h steps, flat prices, no grid limits"""
    T = len(ft)
    ts = TimeSeriesData(
        dt=[3600] * T, ft=ft, gt=gt,
        p_N=[0.30e-3] * T, p_E=[0.08e-3] * T,
    )
    bat = BatteryConfig(
        charge_from_grid=False, discharge_to_grid=False,
        s_capacity=s_max, s_min=0, s_max=s_max, s_initial=0,
        c_min=0, c_max=c_max, d_max=0, p_a=0.25e-3,
    )
    opt = Optimizer(
        strategy=OptimizationStrategy(charging_strategy, 'none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[bat], time_series=ts, eta_c=1.0, eta_d=1.0,
    )
    res = opt.solve()
    assert res['status'] == 'Optimal'
    return res


@pytest.mark.parametrize('strategy', ['attenuate_feedin_peaks', 'attenuate_grid_peaks'])
def test_feedin_profile_is_leveled_below_the_peak(strategy):
    """
    The surplus in the first step is far beyond the charging power, so the export peak is pinned
    there and charging the remaining 170 Wh anywhere in steps 1..4 leaves that peak untouched.
    Capping the peak alone would therefore accept an arbitrary split; the profile has to come out
    leveled instead.
    """
    res = _optimize(strategy, ft=[1000, 100, 100, 100, 100], gt=[0] * 5, c_max=50, s_max=220)

    # peak is pinned by the first step: 1000 Wh surplus less the 50 Wh the battery can take
    assert res['grid_export'][0] == pytest.approx(950, abs=1e-3)
    # remaining 170 Wh charged in equal shares -> equal export in every following step
    assert res['grid_export'][1:] == pytest.approx([100 - 170 / 4] * 4, abs=1e-3)


def test_demand_strategy_leaves_feedin_alone():
    """attenuate_demand_peaks levels grid import only, so the export side keeps the free choice"""
    res = _optimize('attenuate_demand_peaks', ft=[1000, 100, 100, 100, 100], gt=[0] * 5,
                    c_max=50, s_max=220)

    # the battery still fills up, but the export profile is not required to be leveled
    assert sum(res['batteries'][0]['charging_power']) == pytest.approx(220, abs=1e-3)
