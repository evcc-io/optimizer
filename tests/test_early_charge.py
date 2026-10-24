import pytest

from optimizer.optimizer import CHARGING_STRATEGIES, PEAK_STRATEGY_SIDES, BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

# flat prices and a battery worth more than the feed-in revenue: filling it is the money answer,
# when to fill it is not decided by money at all. Four steps of surplus, room for two of them.
STEPS = 4
SURPLUS = 2000.0
P_N = 0.0003
P_E = 0.0001
P_A = 0.0004


# the import side needs a longer horizon than the export side to show anything: with exactly two
# steps of room in four the schedule is pinned either way, and only a horizon with slack leaves the
# timing of the import open at all
GRID_STEPS = 8


def build(charging_strategy, charge_early=True, ft=(SURPLUS,) * STEPS, s_max=4000, charge_from_grid=False):
    steps = len(ft)
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy=charging_strategy, discharging_strategy='none',
                                      charge_early=charge_early),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=charge_from_grid, discharge_to_grid=False,
                                 s_capacity=2.5 * s_max, s_min=0, s_max=s_max, s_initial=0,
                                 c_min=0, c_max=SURPLUS, d_max=0, p_a=P_A)],
        time_series=TimeSeriesData(dt=[3600] * steps, gt=[0] * steps, ft=list(ft),
                                   p_N=[P_N] * steps, p_E=[P_E] * steps),
        eta_c=1.0, eta_d=1.0, M=1e6)


def build_grid_only(charging_strategy, charge_early=True):
    """No solar, so charging comes from grid import only and prc_e_early cannot reach it."""
    return build(charging_strategy, charge_early, ft=(0,) * GRID_STEPS, charge_from_grid=True)


def build_mixed(charging_strategy, charge_early=True):
    """
    Early solar covering half the room, the rest to come from the grid: the only case where the
    two sides pull against each other, because charging the whole battery early needs import that
    a leveled import profile spreads out.
    """
    return build(charging_strategy, charge_early, ft=(SURPLUS, SURPLUS) + (0,) * (GRID_STEPS - 2),
                 s_max=8000, charge_from_grid=True)


def economics(result):
    """s0-insensitive real money: import cost, export revenue, final battery value."""
    battery = result['batteries'][0]
    return (- sum(gi * P_N for gi in result['grid_import'])
            + sum(ge * P_E for ge in result['grid_export'])
            + battery['state_of_charge'][-1] * P_A)


@pytest.mark.parametrize('charging_strategy', ['none', 'attenuate_demand_peaks'])
def test_the_surplus_charges_before_it_exports(charging_strategy):
    """
    none plus the option is what charge_before_export meant. attenuate_demand_peaks rates every
    schedule with the same maximum equally and nothing here draws from the grid, so without the
    option the surplus went out first and the battery took the last two steps.
    """
    result = build(charging_strategy).solve()

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
    assert economics(build('attenuate_demand_peaks').solve()) == \
        pytest.approx(economics(build('none', charge_early=False).solve()))


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
    assert economics(build_grid_only('attenuate_feedin_peaks').solve()) == \
        pytest.approx(economics(build_grid_only('none', charge_early=False).solve()))


@pytest.mark.parametrize('charging_strategy', ['none', *PEAK_STRATEGY_SIDES])
def test_earliness_only_outbids_an_export_side_the_strategy_does_not_level(charging_strategy):
    """
    Filling the battery sooner is not free on either grid side. A leveled export side keeps its
    peak and the tie break stays below the peak weight, an unleveled one has no peak worth
    protecting and earliness takes it outright. Import always stays at the tie break weight,
    since pulling import forward buys a different price and a larger weight trades money for it.
    Pinned on the weights themselves because the schedules depend on the request.
    """
    model = build(charging_strategy)

    assert model.prc_n_early < model.prc_p_peak, "import earliness must stay a tie break"
    if 'exp' in model.peak_sides:
        assert model.prc_e_early < model.prc_p_peak, "exp is leveled, earliness must not outbid it"
    else:
        assert model.prc_e_early > model.prc_n_early, "exp is not leveled, earliness should take it"


def test_solar_fills_the_battery_at_once_while_a_leveled_import_stays_flat():
    """
    Both sides at once on attenuate_demand_peaks: the surplus goes into the battery the moment it
    arrives, because nothing levels the feed-in side, and the grid energy that still has to follow
    it stays spread evenly, because the import side is the one being leveled. Earliness spends the
    peak nobody asked it to shave and leaves the other alone.
    """
    result = build_mixed('attenuate_demand_peaks').solve()

    assert result['status'] == 'Optimal'
    charging = result['batteries'][0]['charging_power']
    assert charging[:2] == pytest.approx([SURPLUS, SURPLUS])
    assert result['grid_export'] == pytest.approx([0.0] * GRID_STEPS)
    # the remaining 4000 Wh arrive as a flat profile over the steps that have no solar left
    assert result['grid_import'][2:] == pytest.approx([4000.0 / (GRID_STEPS - 2)] * (GRID_STEPS - 2))
    assert result['grid_import'][:2] == pytest.approx([0.0, 0.0])


def test_without_the_option_nothing_decides_when_the_battery_fills():
    """
    charge_early is what asks for the early charge, not the charging strategy: a peak strategy on
    its own says how high the grid profile may go and nothing about when to charge, so the schedule
    is free to sit on an empty battery again. Guards against the option being quietly implied.
    """
    charging = build('attenuate_demand_peaks', charge_early=False).solve()['batteries'][0]['charging_power']
    # the surplus leaves first and the battery takes the tail, the schedule none returns
    assert charging[:2] == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize('charging_strategy', CHARGING_STRATEGIES)
def test_the_option_combines_with_every_charging_strategy(charging_strategy):
    """it is orthogonal, so no value of charging_strategy can refuse it"""
    result = build(charging_strategy).solve()

    assert result['status'] == 'Optimal'
    assert sum(result['batteries'][0]['charging_power']) > 0
