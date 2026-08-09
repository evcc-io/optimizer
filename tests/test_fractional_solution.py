import pulp
import pytest
from test_objective_split import build

from optimizer.optimizer import INTEGRALITY_TOLERANCE

# small enough to solve in well under a second, and it has binaries to go fractional: the c_min
# gate of the early charging case is exactly the rule a relaxed binary stops enforcing
CASE = '012-early-charging-not-perfect'


def binaries(optimizer):
    # pulp stores a binary as an integer bounded to 0 and 1, LpBinary never survives on a variable
    return [var for var in optimizer.problem.variables() if var.cat == pulp.LpInteger]


def relax(optimizer, value=0.3):
    """Scribble a relaxation over the binaries, the way a stage that found no integer solution
    leaves them behind."""
    for var in binaries(optimizer):
        var.varValue = value


def test_a_preference_stage_without_an_integer_solution_is_not_kept(monkeypatch):
    # the second stage decides whether to keep its result by reading the variables, and a solver
    # that ran out of clock before it found an integer solution leaves the relaxation in them. That
    # point scores better on the preferences than any real schedule, because it is one the model
    # forbids, so it has to be refused on the status rather than on its score.
    optimizer = build(CASE)
    optimizer.settings.probe_seconds = 0
    optimizer.create_model()

    real_solve = optimizer.problem.solve
    calls = []

    def solve(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:          # the cost stage, left alone
            return real_solve(*args, **kwargs)
        relax(optimizer)             # the preference stage, out of time and empty handed
        optimizer.problem.status = pulp.LpStatusNotSolved
        optimizer.problem.sol_status = pulp.LpSolutionNoSolutionFound
        return optimizer.problem.status

    monkeypatch.setattr(optimizer.problem, 'solve', solve)
    optimizer.solve()

    assert len(calls) == 2, f'the preference stage did not run, {len(calls)} solves'
    assert optimizer.preference_stage.endswith('kept the first stage'), \
        f'preference stage ended as {optimizer.preference_stage}'
    for var in binaries(optimizer):
        assert min(abs(var.varValue), abs(var.varValue - 1)) <= INTEGRALITY_TOLERANCE, \
            f'{var.name} came back at {var.varValue}'


def test_a_fractional_solution_is_reported_as_no_schedule(monkeypatch):
    # the last line of defence, standing in for every stage above deciding correctly. A schedule
    # that breaks the model is worse than no schedule: it looks like an answer and the caller
    # charges a battery by it. Here the solve is faked wholesale, a solver claiming a solution it
    # does not have.
    optimizer = build(CASE)
    optimizer.create_model()

    def probe_then_split(tmpdir, deadline):
        relax(optimizer)
        optimizer.problem.status = pulp.LpStatusOptimal
        optimizer.problem.sol_status = pulp.LpSolutionIntegerFeasible

    monkeypatch.setattr(optimizer, '_probe_then_split', probe_then_split)
    result = optimizer.solve()

    assert result['status'] == 'Not Solved', f"reported {result['status']}"
    assert result['objective_value'] is None
    assert result['batteries'] == []


@pytest.mark.parametrize('value', [0.0, 1.0, INTEGRALITY_TOLERANCE / 2, 1 - INTEGRALITY_TOLERANCE / 2])
def test_a_solution_on_the_integers_passes(value):
    # the guard must not fire on CBC's own rounding, which it reports within its integer tolerance
    optimizer = build(CASE)
    optimizer.create_model()
    relax(optimizer, value)
    assert optimizer._is_integral()
