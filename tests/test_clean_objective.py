import pytest

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

P_N = 0.0002
P_E = 0.0001
P_A = 0.0004
S_INITIAL = 1000.0


def build(dt, gt=None, ft=None, s_initial=S_INITIAL, s_max=5000.0, c_max=1000.0, d_max=0.0):
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy='none', discharging_strategy='none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=True, discharge_to_grid=True,
                                 s_capacity=5000, s_min=0, s_max=s_max, s_initial=s_initial,
                                 c_min=0, c_max=c_max, d_max=d_max, p_a=P_A)],
        time_series=TimeSeriesData(dt=dt, gt=gt or [0] * len(dt), ft=ft or [0] * len(dt),
                                   p_N=[P_N] * len(dt), p_E=[P_E] * len(dt)),
        eta_c=0.95, eta_d=0.95, M=1e6)


def economic_value(result, s_initial=S_INITIAL):
    '''the terms the reported objective value is meant to carry, taken from the schedule'''
    return (- sum(result['grid_import']) * P_N
            + sum(result['grid_export']) * P_E
            + (result['batteries'][0]['state_of_charge'][-1] - s_initial) * P_A)


def test_stored_energy_is_valued_against_the_initial_soc():
    # stored energy is worth more than importing costs, so the battery charges in every step,
    # including the first one. s[0] already carries that first step, which is why the reported
    # value has to reference bat.s_initial instead
    result = build(dt=[3600, 3600]).solve()

    assert result['batteries'][0]['charging_power'][0] > 0, 'the first step must charge for this test to bite'
    assert result['objective_value'] == pytest.approx(economic_value(result))


def test_stored_energy_value_survives_a_discharging_first_step():
    # the same, in the other direction: a battery starting above its maximum soc is discharged
    # right away, so s[0] is below s_initial rather than above it
    result = build(dt=[3600, 3600], s_initial=4000, s_max=2000, c_max=0.0, d_max=1000.0).solve()

    assert result['batteries'][0]['discharging_power'][0] > 0, 'the first step must discharge for this test to bite'
    assert result['objective_value'] == pytest.approx(economic_value(result, s_initial=4000))
