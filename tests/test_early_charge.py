import pytest

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

# flat prices and a battery worth more than the feed-in revenue: filling it is the money answer,
# when to fill it is not decided by money at all. Four steps of surplus, room for two of them.
STEPS = 4
SURPLUS = 2000.0
P_N = 0.0003
P_E = 0.0001
P_A = 0.0004


def build(charging_strategy):
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy=charging_strategy, discharging_strategy='none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=False, discharge_to_grid=False,
                                 s_capacity=10000, s_min=0, s_max=4000, s_initial=0,
                                 c_min=0, c_max=SURPLUS, d_max=0, p_a=P_A)],
        time_series=TimeSeriesData(dt=[3600] * STEPS, gt=[0] * STEPS, ft=[SURPLUS] * STEPS,
                                   p_N=[P_N] * STEPS, p_E=[P_E] * STEPS),
        eta_c=1.0, eta_d=1.0, M=1e6)


# the import side needs a longer horizon than the export side to show anything: with exactly two
# steps of room in four the schedule is pinned either way, and only a horizon with slack leaves the
# timing of the import open at all
GRID_STEPS = 8


def build_grid_only(charging_strategy):
    """
    No solar at all, so charging can only come from grid import and there is nothing to export.
    prc_e_early cannot reach this case: e[t] is zero whatever the schedule does.
    """
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy=charging_strategy, discharging_strategy='none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=True, discharge_to_grid=False,
                                 s_capacity=10000, s_min=0, s_max=4000, s_initial=0,
                                 c_min=0, c_max=SURPLUS, d_max=0, p_a=P_A)],
        time_series=TimeSeriesData(dt=[3600] * GRID_STEPS, gt=[0] * GRID_STEPS, ft=[0] * GRID_STEPS,
                                   p_N=[P_N] * GRID_STEPS, p_E=[P_E] * GRID_STEPS),
        eta_c=1.0, eta_d=1.0, M=1e6)


def economics(result):
    """s0-insensitive real money: import cost, export revenue, final battery value."""
    battery = result['batteries'][0]
    return (- sum(gi * P_N for gi in result['grid_import'])
            + sum(ge * P_E for ge in result['grid_export'])
            + battery['state_of_charge'][-1] * P_A)


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
        values[strategy] = economics(build(strategy).solve())

    assert values['attenuate_demand_peaks'] == pytest.approx(values['none'])


def test_a_battery_charging_from_the_grid_fills_early_when_nothing_levels_the_import():
    """
    prc_e_early can only move a schedule by deferring export, so it cannot reach a battery that
    charges purely from the grid: e[t] is zero whatever the timing. attenuate_feedin_peaks levels
    the feed-in side only, which leaves the import profile entirely undecided, and the charge
    landed in scattered late steps with the battery empty in between. prc_n_early penalizes import
    that lands late, which is the only lever that reaches this case.
    """
    result = build_grid_only('attenuate_feedin_peaks').solve()

    assert result['status'] == 'Optimal'
    charging = result['batteries'][0]['charging_power']
    assert charging[:2] == pytest.approx([SURPLUS, SURPLUS])
    assert charging[2:] == pytest.approx([0.0] * (GRID_STEPS - 2))


def test_leveling_the_demand_side_keeps_the_last_word():
    """
    The counterpart on the side the strategy actually levels: a flat import profile is the lowest
    import peak there is, so attenuate_demand_peaks spreads the same energy over the whole horizon
    rather than taking it in the first two steps. The tie break is correctly too weak to buy
    earliness with peak, exactly as on the feed-in side.
    """
    result = build_grid_only('attenuate_demand_peaks').solve()

    assert result['status'] == 'Optimal'
    assert result['grid_import'] == pytest.approx([4000.0 / GRID_STEPS] * GRID_STEPS)


def test_the_import_side_tie_break_stays_cost_neutral():
    """same as the export side: it picks between schedules, it does not pay for the early charge"""
    values = {}
    for strategy in ('none', 'attenuate_feedin_peaks'):
        values[strategy] = economics(build_grid_only(strategy).solve())

    assert values['attenuate_feedin_peaks'] == pytest.approx(values['none'])
