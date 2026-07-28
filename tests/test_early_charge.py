import pulp
import pytest

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

# flat prices and a battery worth more than the feed-in revenue: filling it is the money answer,
# when to fill it is not decided by money at all. Four steps of surplus, room for two of them.
STEPS = 4
SURPLUS = 2000.0


def build(charging_strategy):
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy=charging_strategy, discharging_strategy='none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=False, discharge_to_grid=False,
                                 s_capacity=10000, s_min=0, s_max=4000, s_initial=0,
                                 c_min=0, c_max=SURPLUS, d_max=0, p_a=0.0004)],
        time_series=TimeSeriesData(dt=[3600] * STEPS, gt=[0] * STEPS, ft=[SURPLUS] * STEPS,
                                   p_N=[0.0003] * STEPS, p_E=[0.0001] * STEPS),
        eta_c=1.0, eta_d=1.0, M=1e6)


def test_a_leveled_import_profile_still_charges_before_it_exports():
    """
    The peak weights rate every schedule with the same maximum and the same ramp equally, and
    nothing here draws from the grid at all, so attenuate_demand_peaks used to leave the whole
    horizon open: the surplus went out first and the battery took the last two steps, the same
    schedule no strategy at all returns. Grid shaping is not a reason to sit on an empty battery.
    """
    result = build('attenuate_demand_peaks').solve()

    assert result['status'] == 'Optimal'
    charging = result['batteries'][0]['charging_power']
    assert charging[:2] == pytest.approx([SURPLUS, SURPLUS])
    assert charging[2:] == pytest.approx([0.0, 0.0])


def test_leveling_the_feed_in_side_keeps_the_last_word():
    """
    The same tie break on attenuate_feedin_peaks, where it collides with the strategy itself:
    charging early leaves the export as two full steps and two empty ones. Leveling wins, the
    export stays flat at half the surplus and the battery fills alongside it.
    """
    result = build('attenuate_feedin_peaks').solve()

    assert result['status'] == 'Optimal'
    assert result['grid_export'] == pytest.approx([SURPLUS / 2] * STEPS)


def test_the_tie_break_stays_cost_neutral():
    """it only picks between schedules, it does not buy the early charge with money"""
    values = {}
    for strategy in ('none', 'attenuate_demand_peaks'):
        model = build(strategy)
        model.solve()
        values[strategy] = pulp.value(model.cost_objective)

    assert values['attenuate_demand_peaks'] == pytest.approx(values['none'])
