import pytest

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData


def build(prc_p_exc_imp=None, p_a=0.0, p_max_imp=500, dt=None, p_N=None, p_E=None):
    dt = dt or [3600]
    p_N = p_N or [0.0003] * len(dt)
    p_E = p_E or [0.0003] * len(dt)
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy='none', discharging_strategy='none'),
        grid=GridConfig(p_max_imp=p_max_imp, p_max_exp=None, prc_p_exc_imp=prc_p_exc_imp),
        batteries=[BatteryConfig(charge_from_grid=False, discharge_to_grid=False,
                                 s_capacity=1000, s_min=500, s_max=1000, s_initial=1000,
                                 c_min=0, c_max=0, d_max=2000, p_a=p_a)],
        time_series=TimeSeriesData(dt=dt, gt=[2000] * len(dt), ft=[0] * len(dt), p_N=p_N, p_E=p_E),
        eta_c=0.95, eta_d=0.95, M=1e6)


def test_penalty_base_matches_import_price_when_nothing_else_dominates():
    model = build(prc_p_exc_imp=None, p_a=0.0)
    assert model.penalty_base == pytest.approx(0.0003)


def test_penalty_base_accounts_for_demand_rate_via_horizon_hours():
    # dt sums to 2 hours, so 1 W of demand rate converts to 0.5 currency/Wh
    model = build(prc_p_exc_imp=1.0, dt=[3600, 3600])
    assert model.penalty_base == pytest.approx(0.5)


def test_penalty_base_accounts_for_final_soc_value():
    model = build(prc_p_exc_imp=None, p_a=0.05)
    assert model.penalty_base == pytest.approx(0.05)


def test_soc_min_floor_holds_against_a_large_demand_rate_incentive():
    """
    Regression test for the penalty_base fix: without folding prc_p_exc_imp into
    penalty_base, a large enough demand rate makes it cheaper for the solver to drain
    the battery below its soft s_min floor than to pay the demand charge - draining to
    s=0 here instead of stopping at s_min=500 (see conversation history / PR description
    for the pre-fix numbers: s ends at 0.0, d at 950.0 instead of 500.0 / 475.0).
    """
    model = build(prc_p_exc_imp=1.0, p_max_imp=500)
    model.create_model()
    result = model.solve()

    assert result['status'] == 'Optimal'
    assert result['batteries'][0]['state_of_charge'][0] == pytest.approx(500.0)
    assert result['batteries'][0]['discharging_power'][0] == pytest.approx(475.0)
    # demand charge still applies to the unavoidable remainder of the exceedance
    assert result['grid_import_overshoot'][0] == pytest.approx(1025.0)
