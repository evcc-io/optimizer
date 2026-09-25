import pulp
import pytest
from test_objective_split import build

# small enough to solve in well under a second, with a strategy so the preference stage runs
CASE = '012-early-charging-not-perfect'


def break_a_row(optimizer):
    """Scribble a schedule that keeps every binary on the integers and breaks the energy balance,
    the way CBC's solution file has come back when the clock ran out inside the root heuristics:
    one battery at full charging power in one step, nothing to pay for it."""
    for var in optimizer.problem.variables():
        var.varValue = 0.
    charge = optimizer.variables['c'][0][1]
    charge.varValue = charge.upBound


def test_a_solution_off_the_rows_is_reported_as_no_schedule(monkeypatch):
    # the last line of defence, standing in for every stage above deciding correctly. The solve
    # is faked wholesale, a solver claiming a proven optimum over values that break the model.
    optimizer = build(CASE)
    optimizer.create_model()

    def probe_then_split(tmpdir, deadline):
        break_a_row(optimizer)
        optimizer.problem.status = pulp.LpStatusOptimal
        optimizer.problem.sol_status = pulp.LpSolutionOptimal

    monkeypatch.setattr(optimizer, '_probe_then_split', probe_then_split)
    result = optimizer.solve()

    assert result['status'] == 'Not Solved', f"reported {result['status']}"
    assert result['objective_value'] is None
    assert result['batteries'] == []


def test_a_cost_stage_off_the_rows_falls_back_to_the_probe(monkeypatch):
    # the split keeps whatever the probe reached when the cost stage comes back unusable, and a
    # vector that breaks the rows is unusable whatever its status says
    optimizer = build(CASE)
    optimizer.create_model()

    real_solve = optimizer.problem.solve
    calls = []

    def solve(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:          # the probe, real, reported unproven so the split runs
            real_solve(*args, **kwargs)
            optimizer.problem.sol_status = pulp.LpSolutionIntegerFeasible
            return optimizer.problem.status
        break_a_row(optimizer)       # the cost stage, a solution file that is not the schedule
        optimizer.problem.status = pulp.LpStatusOptimal
        optimizer.problem.sol_status = pulp.LpSolutionIntegerFeasible
        return optimizer.problem.status

    monkeypatch.setattr(optimizer.problem, 'solve', solve)
    result = optimizer.solve()

    assert len(calls) == 2, f'{len(calls)} solves'
    assert optimizer.solve_path == 'split, kept the probe', optimizer.solve_path
    assert result['status'] == 'Feasible', f"reported {result['status']}"
    assert optimizer._is_feasible()


def test_a_preference_stage_off_the_rows_is_not_kept(monkeypatch):
    # the second stage decides whether to keep its result by reading the variables. A vector that
    # breaks the rows scores better on the preferences than any real schedule, so it has to be
    # refused on the rows rather than on its score.
    optimizer = build(CASE)
    optimizer.settings.probe_seconds = 0
    optimizer.create_model()

    real_solve = optimizer.problem.solve
    calls = []

    def solve(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:          # the cost stage, left alone
            return real_solve(*args, **kwargs)
        break_a_row(optimizer)       # the preference stage
        optimizer.problem.status = pulp.LpStatusOptimal
        optimizer.problem.sol_status = pulp.LpSolutionOptimal
        return optimizer.problem.status

    monkeypatch.setattr(optimizer.problem, 'solve', solve)
    result = optimizer.solve()

    assert len(calls) == 2, f'the preference stage did not run, {len(calls)} solves'
    assert optimizer.preference_stage.endswith('kept the first stage'), \
        f'preference stage ended as {optimizer.preference_stage}'
    assert result['status'] in ('Optimal', 'Feasible'), f"reported {result['status']}"
    assert optimizer._is_feasible()


@pytest.mark.parametrize('case', [CASE, '010-infesible-charge-goal', '020-weird-charging-at-night'])
def test_a_solution_on_the_rows_passes(case):
    # the guard must not fire on CBC's own rounding: it writes 8 significant digits, so the
    # balance rows of a large battery come back off by a fraction of a Wh
    optimizer = build(case)
    optimizer.create_model()
    optimizer.problem.solve(optimizer._solver('/tmp'))
    assert optimizer._is_feasible()
