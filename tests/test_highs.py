"""HiGHS as the solver (OPTIMIZER_SOLVER=highs): the stored optima, the cost stage gap and the cutoff."""
import json
import pathlib
from tempfile import TemporaryDirectory

import pulp
import pytest
from test_objective_split import build

from optimizer.settings import OptimizerSettings

CASES = sorted(p.stem for p in pathlib.Path('test_cases').glob('*.json'))


def highs(case, **settings):
    optimizer = build(case)
    optimizer.settings = OptimizerSettings(solver='highs', time_limit=10, **settings)
    return optimizer


@pytest.mark.parametrize('case', CASES)
def test_stored_optimum_without_gap(case):
    # the stored objectives are proven optima. At the production gap HiGHS may stop anywhere
    # within it, as CBC does, so compare where neither may stop early. The whole time limit goes
    # to the joint solve: a slow runner past the default probe share falls back to the split, whose
    # cost stage is capped at COST_TIME_LIMIT and returns a feasible, unproven schedule
    expected = json.loads(pathlib.Path(f'test_cases/{case}.json').read_text()).get('expected_response')
    if expected is None or 'objective_value' not in expected:
        pytest.skip('no stored objective')
    result = highs(case, gap_abs=None, gap_share=0, probe_seconds=10).solve()

    assert result['status'] == expected['status']
    # a hundredth of a cent, what the tie break gives away to the solver's rounding
    assert result['objective_value'] == pytest.approx(expected['objective_value'], abs=1e-4, rel=1e-5)


def test_cost_stage_gap_comes_from_the_solver_model():
    # pulp 4 leaves best_bound unset for HiGHS, so the gap is read from HiGHS itself
    optimizer = highs('026-attenuate-grid-peaks', probe_seconds=0)
    optimizer.solve()

    assert optimizer.solve_path == 'split'
    assert optimizer.cost_stage_gap == 0  # proven, nothing left open


def test_overrunning_solve_is_interrupted_and_keeps_its_incumbent():
    # a long horizon HiGHS needs about 3 s for, cut off after a tenth of a second
    optimizer = highs('028-attenuate-grid-peaks-long-horizon')
    optimizer.create_model()

    with TemporaryDirectory() as tmpdir, optimizer._hard_limit(0.1) as cut:
        stats = optimizer.problem.solve(optimizer._solver(tmpdir, timeLimit=10))

    assert cut.is_set()
    assert stats.status == pulp.LpSolveStatus.Interrupted
    # an incumbent, if HiGHS had one, comes back usable rather than as noise
    assert not stats.has_solution or optimizer._is_feasible()


def test_solve_finishing_in_time_is_left_alone():
    optimizer = highs('012-early-charging-not-perfect')
    optimizer.create_model()

    with TemporaryDirectory() as tmpdir, optimizer._hard_limit(5) as cut:
        stats = optimizer.problem.solve(optimizer._solver(tmpdir, timeLimit=10))

    assert not cut.is_set()
    assert stats.status in (pulp.LpSolveStatus.Optimal, pulp.LpSolveStatus.GapLimit)
