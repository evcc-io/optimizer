"""The hybrid solver (OPTIMIZER_SOLVER=hybrid): CBC proves the joint probe, HiGHS runs the split and continuity."""
import json
import pathlib
from tempfile import TemporaryDirectory

import pulp
import pytest
from test_objective_split import build

from optimizer.optimizer import Optimizer
from optimizer.settings import OptimizerSettings

CASES = sorted(p.stem for p in pathlib.Path('test_cases').glob('*.json'))


def hybrid(case, **settings):
    optimizer = build(case)
    optimizer.settings = OptimizerSettings(solver='hybrid', time_limit=10, **settings)
    return optimizer


@pytest.fixture
def solvers(monkeypatch):
    """Solver class per stage, in the order the stages ran."""
    seen = []
    original = Optimizer._solver

    def record(self, tmpdir, **options):
        solver = original(self, tmpdir, **options)
        seen.append((self._stage, type(solver)))
        return solver

    monkeypatch.setattr(Optimizer, '_solver', record)
    return seen


@pytest.mark.parametrize('case', CASES)
def test_stored_optimum_without_gap(case):
    # as test_highs: proven optima compared where neither solver may stop early, the whole limit
    # on the joint solve so a slow runner does not fall into the capped split
    expected = json.loads(pathlib.Path(f'test_cases/{case}.json').read_text()).get('expected_response')
    if expected is None or 'objective_value' not in expected:
        pytest.skip('no stored objective')
    result = hybrid(case, gap_abs=None, gap_share=0, probe_seconds=10).solve()

    assert result['status'] == expected['status']
    assert result['objective_value'] == pytest.approx(expected['objective_value'], abs=1e-4, rel=1e-5)


def test_split_runs_on_highs(solvers):
    optimizer = hybrid('026-attenuate-grid-peaks', probe_seconds=0)
    optimizer.solve()

    assert optimizer.solve_path == 'split'
    stages = dict(solvers)
    assert stages['cost'] is pulp.HiGHS
    assert stages['tie_break'] is pulp.HiGHS
    # the gap of a HiGHS cost stage comes from the stats, not a CBC log
    assert optimizer.cost_stage_gap == 0


def test_joint_path_is_all_cbc(solvers):
    optimizer = hybrid('012-early-charging-not-perfect')
    optimizer.solve()

    assert optimizer.solve_path == 'joint'
    assert {cls for stage, cls in solvers if stage == 'probe'} == {pulp.COIN_CMD}


def test_hard_limit_follows_the_stage():
    optimizer = hybrid('012-early-charging-not-perfect')
    optimizer.create_model()

    # the CBC probe is left to its -sec, no interrupt is armed
    with optimizer._timed('probe'), optimizer._hard_limit(5):
        assert optimizer._cutoff is None
    # a HiGHS stage is armed and interrupted past its limit, keeping its incumbent
    with TemporaryDirectory() as tmpdir, optimizer._timed('continuity'), optimizer._hard_limit(5) as cut:
        assert optimizer._cutoff is not None
        stats = optimizer.problem.solve(optimizer._solver(tmpdir, timeLimit=10))
    assert not cut.is_set()
    assert stats.status in (pulp.LpSolveStatus.Optimal, pulp.LpSolveStatus.GapLimit)
