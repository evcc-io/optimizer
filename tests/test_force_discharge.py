import pytest

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData


def build(d_demand, discharge_to_grid=True, s_initial=10000, s_min=2000, d_max=5000, p_E=None):
    """Four idle hours: no house load, no solar, no reason to move energy on price alone."""
    dt = [3600] * 4
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy='none', discharging_strategy='none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=False, discharge_to_grid=discharge_to_grid,
                                 s_capacity=10000, s_min=s_min, s_max=10000, s_initial=s_initial,
                                 c_min=0, c_max=5000, d_max=d_max, p_a=0.0003,
                                 d_demand=d_demand)],
        time_series=TimeSeriesData(dt=dt, gt=[0] * 4, ft=[0] * 4,
                                   p_N=[0.0003] * 4, p_E=p_E or [0.0001] * 4),
        eta_c=0.95, eta_d=0.95, M=1e6)


def test_no_discharge_demand_leaves_the_battery_alone():
    # baseline: p_a beats the export price, so nothing moves without a demand
    result = build(None).solve()

    assert result['status'] == 'Optimal'
    assert result['batteries'][0]['discharging_power'] == pytest.approx([0, 0, 0, 0])


def test_discharge_demand_forces_export_against_the_price():
    # the same horizon, now with a demand in steps 1 and 2 - it is served even though selling at
    # p_E is worth less than keeping the energy at p_a
    result = build([0, 2000, 3000, 0]).solve()

    assert result['status'] == 'Optimal'
    assert result['batteries'][0]['discharging_power'] == pytest.approx([0, 2000, 3000, 0])
    assert result['grid_export'] == pytest.approx([0, 2000, 3000, 0])


def test_discharge_demand_is_clipped_to_d_max():
    result = build([4000, 0, 0, 0], d_max=1000).solve()

    assert result['batteries'][0]['discharging_power'][0] == pytest.approx(1000)


def test_discharge_demand_stops_at_the_s_min_reserve():
    # demanding more than the reserve allows must not drain through s_min: the SoC penalty
    # outweighs the unserved demand penalty, so the remainder is simply given up
    result = build([5000, 5000, 0, 0], s_initial=4000, s_min=2000).solve()

    assert result['status'] == 'Optimal'
    soc = result['batteries'][0]['state_of_charge']
    assert min(soc) == pytest.approx(2000)
    # 2000 Wh of usable content, delivered through the discharge efficiency
    assert sum(result['batteries'][0]['discharging_power']) == pytest.approx(2000 * 0.95)


def test_discharge_demand_is_soft_when_the_grid_is_the_only_sink():
    # without discharge_to_grid there is nowhere for the energy to go, and the request stays
    # solvable instead of turning infeasible
    result = build([3000, 0, 0, 0], discharge_to_grid=False).solve()

    assert result['status'] == 'Optimal'
    assert result['batteries'][0]['discharging_power'][0] == pytest.approx(0)
