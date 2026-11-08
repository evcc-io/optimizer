"""The MIP start handover: every stage after the probe starts from the schedule the stage before it reached."""
import pytest
from test_objective_split import build

from optimizer.optimizer import Optimizer, cbc_takes_mip_start
from optimizer.settings import OptimizerSettings

CASE = '023-c_min-limit-kept-with-p_demand-set'
SOLVER = Optimizer._solver
SOLVERS = ['hybrid', 'cbc']


def split(monkeypatch, solver, cold=False):
    """The split, with the start handed on or stripped; returns the result and the (stage, start) calls."""
    starts = []

    def record(self, tmpdir, **options):
        starts.append((self._stage, options.get('warmStart', False)))
        if cold:
            options.pop('warmStart', None)
        return SOLVER(self, tmpdir, **options)

    monkeypatch.setattr(Optimizer, '_solver', record)
    optimizer = build(CASE)
    # the tie break gets a tenth of the limit; 10 s left it 1 s, which a loaded CI runner overran
    optimizer.settings = OptimizerSettings(solver=solver, time_limit=60, probe_seconds=0, gap_abs=None, gap_share=0)
    return optimizer.solve(), starts


@pytest.mark.parametrize('solver', SOLVERS)
def test_tie_break_starts_from_the_cost_stage(monkeypatch, solver):
    result, starts = split(monkeypatch, solver)

    assert result['status'] == 'Optimal'
    assert ('tie_break', True) in starts


@pytest.mark.parametrize('solver', SOLVERS)
def test_started_split_matches_the_cold_one(monkeypatch, solver):
    warm, _ = split(monkeypatch, solver)
    cold, _ = split(monkeypatch, solver, cold=True)

    assert warm['status'] == cold['status'] == 'Optimal'
    assert warm['objective_value'] == pytest.approx(cold['objective_value'], abs=1e-4, rel=1e-5)


@pytest.mark.parametrize('version, takes', [('2.10.12', False), ('2.10.13', True), ('2.11.0', True)])
def test_cbc_takes_a_mip_start_from_2_10_13(tmp_path, version, takes):
    cbc = tmp_path / 'cbc'
    cbc.write_text(f'#!/bin/sh\necho "Welcome to the CBC MILP Solver"\necho "Version: {version}"\n')
    cbc.chmod(0o755)

    assert cbc_takes_mip_start(str(cbc)) is takes


def test_an_old_cbc_gets_no_start(monkeypatch, tmp_path):
    monkeypatch.setattr('optimizer.optimizer.cbc_takes_mip_start', lambda path: False)
    optimizer = build(CASE)
    optimizer.settings = OptimizerSettings(solver='cbc')

    assert 'warmStart' not in SOLVER(optimizer, str(tmp_path), warmStart=True).optionsDict
