import numpy as np

from optimizer.optimizer import (
    BatteryConfig,
    GridConfig,
    OptimizationStrategy,
    Optimizer,
    TimeSeriesData,
)


def _solve(withhold_charge: bool):
    """
    Cost-neutral scenario: storing energy is worth exactly the export price (p_a == p_E),
    so the attenuate_grid_peaks strategy is the only term deciding *whether* and *when* the
    battery absorbs surplus. Surplus is available in a small morning hump (t0, ft=100) and a
    large midday peak (t2, ft=1000). The morning hump sits at t1 (not t0) so the first-step
    SOC baseline used by the clean objective stays identical across both runs.
    """
    strategy = OptimizationStrategy(
        charging_strategy='attenuate_grid_peaks',
        discharging_strategy='none',
    )
    grid = GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None)
    battery = BatteryConfig(
        charge_from_grid=False,
        discharge_to_grid=False,
        s_capacity=2000,
        s_min=0,
        s_max=2000,
        s_initial=0,
        c_min=0,
        c_max=5000,
        d_max=0,
        p_a=0.1,
        withhold_charge=withhold_charge,
    )
    time_series = TimeSeriesData(
        dt=[3600, 3600, 3600, 3600],
        gt=[0, 0, 0, 0],
        ft=[0, 100, 1000, 0],
        p_N=[0.3, 0.3, 0.3, 0.3],
        p_E=[0.1, 0.1, 0.1, 0.1],
    )

    optimizer = Optimizer(strategy, grid, [battery], time_series, eta_c=1.0, eta_d=1.0)
    return optimizer.solve()


def test_defer_charges_all_available_surplus():
    """Without the withhold capability the strategy only re-times charging and still stores
    the low-solar morning surplus."""
    result = _solve(withhold_charge=False)

    assert result['status'] == 'Optimal'
    charging = result['batteries'][0]['charging_power']
    assert np.isclose(charging[1], 100, atol=1e-3), "morning surplus should be stored"
    assert np.isclose(charging[2], 1000, atol=1e-3), "midday peak should be stored"


def test_withhold_skips_low_solar_charging():
    """With the withhold capability the strategy forgoes charging below the solar peak,
    leaving capacity free and exporting the morning surplus instead."""
    result = _solve(withhold_charge=True)

    assert result['status'] == 'Optimal'
    charging = result['batteries'][0]['charging_power']
    assert np.isclose(charging[1], 0, atol=1e-3), "morning surplus should be withheld, not stored"
    assert np.isclose(charging[2], 1000, atol=1e-3), "midday peak should still be stored"


def test_withhold_is_cost_neutral():
    """Withholding must not change the actual economic outcome; it only acts on cost-neutral
    choices (the clean objective excludes strategy incentives)."""
    defer = _solve(withhold_charge=False)
    withhold = _solve(withhold_charge=True)

    assert np.isclose(defer['objective_value'], withhold['objective_value'], rtol=1e-6, atol=1e-9)
