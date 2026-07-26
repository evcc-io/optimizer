import pytest

from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

M = 1e6
DT = [3600, 900]
GT = [4000.0, 500.0]
FT = [1500.0, 250.0]
C_MAX = 2000
D_MAX = 3000


def build(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None):
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy='none', discharging_strategy='none'),
        grid=GridConfig(p_max_imp=p_max_imp, p_max_exp=p_max_exp, prc_p_exc_imp=prc_p_exc_imp),
        batteries=[BatteryConfig(charge_from_grid=True, discharge_to_grid=True,
                                 s_capacity=10000, s_min=0, s_max=10000, s_initial=5000,
                                 c_min=0, c_max=C_MAX, d_max=D_MAX, p_a=0.0004)],
        time_series=TimeSeriesData(dt=DT, gt=GT, ft=FT,
                                   p_N=[0.0003] * len(DT), p_E=[0.0001] * len(DT)),
        eta_c=0.95, eta_d=0.95, M=M)


def switch_coefficient(model, excess, binary):
    """The big-M coefficient of the constraint tying an excess variable to its switch."""
    found = []
    for constraint in model.problem.constraints.values():
        names = {v.name for v in constraint.keys()}
        if excess in names and binary in names:
            found.append(constraint[next(v for v in constraint.keys() if v.name == binary)])
    assert len(found) == 1, f'expected one switch constraint, found {len(found)}'
    return found[0]


@pytest.mark.parametrize('prc_p_exc_imp', [None, 0.01])
def test_import_excess_switch_is_capped_at_what_the_step_can_absorb(prc_p_exc_imp):
    # This coefficient is what the LP relaxation sees. At the global M it can open an excess
    # three orders beyond anything a schedule could use, which leaves the bound useless and the
    # branch and bound searching for an answer it already has. Demand plus grid charge capacity
    # is the real ceiling and excludes no integer point.
    model = build(p_max_imp=1000, prc_p_exc_imp=prc_p_exc_imp)
    model.create_model()

    for t, dt in enumerate(DT):
        coefficient = switch_coefficient(model, f'p_imp_pen_{t}', f'z_imp_lim_{t}')
        assert coefficient == pytest.approx(GT[t] + C_MAX * dt / 3600.)
        assert coefficient < M


def test_reported_overshoot_is_the_import_beyond_the_limit():
    # the switch still orders the two import portions, so the excess variable carries exactly
    # what went over the limit rather than an arbitrary share of the import
    result = build(p_max_imp=1000, prc_p_exc_imp=0.01).solve()

    assert result['status'] == 'Optimal'
    for t, dt in enumerate(DT):
        over = max(0.0, result['grid_import'][t] - 1000 * dt / 3600.)
        assert result['grid_import_overshoot'][t] == pytest.approx(over, abs=1e-6)
