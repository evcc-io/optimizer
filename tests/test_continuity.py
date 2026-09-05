import time
from tempfile import TemporaryDirectory

import numpy as np
import pulp
import pytest

from optimizer.continuity import minimize_interruptions
from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData


def build(strategy: str = 'none') -> Optimizer:
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy=strategy, discharging_strategy='none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(charge_from_grid=True, discharge_to_grid=False,
                                 s_capacity=5000, s_min=0, s_max=5000, s_initial=0,
                                 c_min=1000, c_max=2000, d_max=0, p_a=0,
                                 s_goal=[0, 0, 0, 0, 0, 1500])],
        time_series=TimeSeriesData(dt=[900] * 6, gt=[0] * 6, ft=[0] * 6,
                                   p_N=[0.001] + [0.0003] * 5, p_E=[0] * 6),
        eta_c=1, eta_d=1,
    )


def starts(charging: list[float]) -> int:
    active = np.array(charging) > 0.01
    return int(np.count_nonzero(active & ~np.r_[False, active[:-1]]))


def seed_fragmented(model: Optimizer, monkeypatch: pytest.MonkeyPatch) -> None:
    original = model._probe_then_split

    def seeded(tmpdir: str, deadline: float | None) -> None:
        for t, energy in enumerate([0, 500, 0, 500, 0, 500]):
            model.problem += model.variables['c'][0][t] == energy, f'seed_{t}'
        original(tmpdir, deadline)
        for t in model.time_steps:
            del model.problem.constraints[f'seed_{t}']

    monkeypatch.setattr(model, '_probe_then_split', seeded)


@pytest.mark.parametrize('probe_seconds', [None, 0])
def test_equal_prices_prefer_one_session(monkeypatch: pytest.MonkeyPatch, probe_seconds: float | None):
    model = build()
    model.settings.probe_seconds = probe_seconds
    seed_fragmented(model, monkeypatch)

    result = model.solve()

    assert result['status'] == 'Optimal'
    assert starts(result['batteries'][0]['charging_power']) == 1
    assert result['batteries'][0]['state_of_charge'][-1] == pytest.approx(1500, abs=0.1)
    assert pulp.value(model.cost_objective) == pytest.approx(-0.45, abs=1e-5)


def test_price_gaps_keep_interruptions(monkeypatch: pytest.MonkeyPatch):
    model = build()
    model.time_series.p_N = [0.001, 0.0003, 0.001, 0.0003, 0.001, 0.0003]
    seed_fragmented(model, monkeypatch)

    result = model.solve()

    assert starts(result['batteries'][0]['charging_power']) == 3
    assert pulp.value(model.cost_objective) == pytest.approx(-0.45, abs=1e-5)


def test_grid_shaping_takes_priority(monkeypatch: pytest.MonkeyPatch):
    model = build('attenuate_demand_peaks')
    model.time_series.gt = [0, 0, 2000, 0, 2000, 0]
    seed_fragmented(model, monkeypatch)

    result = model.solve()

    assert starts(result['batteries'][0]['charging_power']) == 3
    assert max(result['grid_import']) == pytest.approx(2000, abs=0.01)


def test_short_first_slot(monkeypatch: pytest.MonkeyPatch):
    model = build()
    model.time_series.dt[0] = 100
    seed_fragmented(model, monkeypatch)

    result = model.solve()

    assert starts(result['batteries'][0]['charging_power']) == 1
    assert result['batteries'][0]['charging_power'][0] == pytest.approx(0, abs=0.01)


def test_repeated_solve_does_not_keep_polishing_constraints(monkeypatch: pytest.MonkeyPatch):
    model = build()
    seed_fragmented(model, monkeypatch)
    model.solve()
    constraints = set(model.problem.constraints)
    variables = {v.name for v in model.problem.variables()}

    result = model.solve()

    assert starts(result['batteries'][0]['charging_power']) == 1
    assert set(model.problem.constraints) == constraints
    assert {v.name for v in model.problem.variables()} == variables


@pytest.mark.parametrize('price', [0, -0.0003])
def test_nonpositive_prices(monkeypatch: pytest.MonkeyPatch, price: float):
    model = build()
    model.time_series.p_N = [0.001] + [price] * 5
    model.batteries[0].s_max = 1500
    seed_fragmented(model, monkeypatch)

    result = model.solve()

    assert starts(result['batteries'][0]['charging_power']) == 1
    assert pulp.value(model.cost_objective) == pytest.approx(-price * 1500, abs=2e-5)


def test_forced_demand_is_preserved(monkeypatch: pytest.MonkeyPatch):
    model = build()
    model.batteries[0].p_demand = [0, 500, 0, 500, 0, 500]
    seed_fragmented(model, monkeypatch)

    result = model.solve()

    assert result['batteries'][0]['charging_power'] == pytest.approx([0, 500, 0, 500, 0, 500], abs=0.01)


@pytest.mark.parametrize('c_min', [0, 1000])
def test_uninterrupted_or_unrestricted_batteries_skip_the_solver(monkeypatch: pytest.MonkeyPatch, c_min: float):
    model = build()
    model.batteries[0].c_min = c_min
    model.batteries[0].s_goal = None
    model.solve()

    def unexpected_solver(*args, **kwargs):
        pytest.fail('continuity should not invoke CBC')

    monkeypatch.setattr(model, '_solver', unexpected_solver)
    with TemporaryDirectory() as tmpdir:
        minimize_interruptions(model, tmpdir, None)


def fragmented_model(monkeypatch: pytest.MonkeyPatch) -> Optimizer:
    model = build()
    seed_fragmented(model, monkeypatch)
    with monkeypatch.context() as context:
        context.setattr('optimizer.optimizer.minimize_interruptions', lambda *args: None)
        model.solve()
    return model


def test_expired_deadline_keeps_incumbent(monkeypatch: pytest.MonkeyPatch):
    model = fragmented_model(monkeypatch)
    solution = {var: var.varValue for var in model.problem.variables()}

    def unexpected_solver(*args, **kwargs):
        pytest.fail('continuity should not invoke CBC after the deadline')

    monkeypatch.setattr(model, '_solver', unexpected_solver)
    with TemporaryDirectory() as tmpdir:
        minimize_interruptions(model, tmpdir, time.monotonic() - 1)

    assert {var: var.varValue for var in model.problem.variables()} == solution


@pytest.mark.parametrize('outcome', ['infeasible', 'fractional', 'error', 'invalid'])
def test_failed_polish_restores_incumbent(monkeypatch: pytest.MonkeyPatch, outcome: str):
    model = fragmented_model(monkeypatch)
    solution = {var: var.varValue for var in model.problem.variables()}
    status = model.problem.status, model.problem.sol_status

    def failed_solve(candidate: pulp.LpProblem, *args, **kwargs):
        for var in candidate.variables():
            var.varValue = 0.3 if var.cat == pulp.LpInteger else 0
        match outcome:
            case 'error':
                raise pulp.PulpSolverError('CBC unavailable')
            case 'infeasible':
                candidate.sol_status = pulp.LpSolutionInfeasible
            case 'fractional':
                candidate.sol_status = pulp.LpSolutionNoSolutionFound
            case 'invalid':
                candidate.sol_status = pulp.LpSolutionOptimal
        return pulp.LpStatusNotSolved

    monkeypatch.setattr(pulp.LpProblem, 'solve', failed_solve)
    with TemporaryDirectory() as tmpdir:
        minimize_interruptions(model, tmpdir, None)

    assert {var: var.varValue for var in model.problem.variables()} == solution
    assert (model.problem.status, model.problem.sol_status) == status


def test_polish_does_not_upgrade_feasible_status(monkeypatch: pytest.MonkeyPatch):
    model = fragmented_model(monkeypatch)
    model.problem.sol_status = pulp.LpSolutionIntegerFeasible

    with TemporaryDirectory() as tmpdir:
        minimize_interruptions(model, tmpdir, None)

    assert starts([pulp.value(v) for v in model.variables['c'][0]]) == 1
    assert model.problem.sol_status == pulp.LpSolutionIntegerFeasible
