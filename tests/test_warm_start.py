"""The MIP start handover: every HiGHS stage after the probe starts from the schedule the stage before it reached."""
import pytest
from test_objective_split import build

from optimizer.optimizer import Optimizer
from optimizer.settings import OptimizerSettings

CASE = '023-c_min-limit-kept-with-p_demand-set'


def split(monkeypatch, cold=False):
    """The split on hybrid, with the start handed on or stripped; returns the result and the (stage, start) calls."""
    starts = []
    original = Optimizer._solver

    def record(self, tmpdir, **options):
        starts.append((self._stage, options.get('warmStart', False)))
        if cold:
            options.pop('warmStart', None)
        return original(self, tmpdir, **options)

    monkeypatch.setattr(Optimizer, '_solver', record)
    optimizer = build(CASE)
    optimizer.settings = OptimizerSettings(solver='hybrid', time_limit=10, probe_seconds=0, gap_abs=None, gap_share=0)
    return optimizer.solve(), starts


def test_tie_break_starts_from_the_cost_stage(monkeypatch):
    result, starts = split(monkeypatch)

    assert result['status'] == 'Optimal'
    assert ('tie_break', True) in starts


def test_started_split_matches_the_cold_one(monkeypatch):
    warm, _ = split(monkeypatch)
    cold, _ = split(monkeypatch, cold=True)

    assert warm['status'] == cold['status'] == 'Optimal'
    assert warm['objective_value'] == pytest.approx(cold['objective_value'], abs=1e-4, rel=1e-5)
