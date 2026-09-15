import pytest

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData


def build(primary_goal):
    # one step, all PV surplus (ft) is above household demand (gt=0): a battery with full
    # headroom (s_initial=0) could take all of it, but p_a is set far below p_E so under
    # minimize_cost selling it now genuinely earns more than storing it does - export should win
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy='none', discharging_strategy='none',
                                      primary_goal=primary_goal),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=False, discharge_to_grid=False,
                                 s_capacity=2000, s_min=0, s_max=2000, s_initial=0,
                                 c_min=0, c_max=2000, d_max=0, p_a=0.00001)],
        time_series=TimeSeriesData(dt=[3600], gt=[0], ft=[2000], p_N=[0.0003], p_E=[0.0003]),
        eta_c=0.95, eta_d=0.95, M=1e6)


def test_minimize_cost_exports_when_storing_is_worth_less_than_selling():
    result = build('minimize_cost').solve()

    assert result['status'] == 'Optimal'
    assert result['grid_export'][0] == pytest.approx(2000.0)
    assert result['batteries'][0]['charging_power'][0] == pytest.approx(0.0)


def test_maximize_self_consumption_charges_instead_of_exporting():
    # same request, only primary_goal differs: exporting is no longer revenue but a cost
    # weighted the same as p_E, so a battery with headroom is preferred over selling even
    # though selling would have earned strictly more money
    result = build('maximize_self_consumption').solve()

    assert result['status'] == 'Optimal'
    assert result['batteries'][0]['charging_power'][0] == pytest.approx(2000.0)
    assert result['grid_export'][0] == pytest.approx(0.0)


def test_primary_goal_defaults_to_minimize_cost():
    # OptimizationStrategy without an explicit primary_goal behaves like minimize_cost,
    # so existing callers that never set the field are unaffected
    strategy = OptimizationStrategy(charging_strategy='none', discharging_strategy='none')
    assert strategy.primary_goal == 'minimize_cost'
