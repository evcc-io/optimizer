"""The stage clock: every solve leaves behind where its wall time went.

The access log only carries the total response time, so stage_seconds is what production
attributes latency with. These tests pin the keys each solve path must produce.
"""
import dataclasses

import pulp
from test_objective_split import build

from optimizer.optimizer import Optimizer
from optimizer.settings import OptimizerSettings


def test_joint_path_times_build_and_probe():
    # a small case the probe proves outright: no split, so no cost or tie break stage ran
    optimizer = build('012-early-charging-not-perfect')
    optimizer.solve()

    assert optimizer.solve_path == 'joint'
    assert set(optimizer.stage_seconds) == {'build', 'probe'}
    # no cost stage ran, so there is no gap to report
    assert optimizer.cost_stage_gap is None
    assert all(seconds >= 0 for seconds in optimizer.stage_seconds.values())


def test_split_path_times_every_stage():
    # probe_seconds=0 forces the split, and this case carries a strategy so the tie break runs
    optimizer = build('026-attenuate-grid-peaks')
    optimizer.settings = OptimizerSettings(probe_seconds=0, time_limit=10)
    optimizer.solve()

    assert optimizer.solve_path == 'split'
    # no probe ran, so no probe key: an absent stage must be absent, not zero
    assert set(optimizer.stage_seconds) == {'build', 'cost', 'tie_break'}
    assert all(seconds >= 0 for seconds in optimizer.stage_seconds.values())
    # the cost stage leaves behind what it found and what CBC could not rule out above it
    assert optimizer.cost_stage_value is not None
    assert optimizer.cost_stage_gap is not None
    assert optimizer.cost_stage_gap >= 0


def test_split_with_cost_stage_stopped_by_the_clock_is_feasible(monkeypatch):
    # the tie break LP always proves itself, which must not turn the cut off money into Optimal
    original = Optimizer._solve

    def cost_stopped(self, solver):
        stats = original(self, solver)
        return dataclasses.replace(stats, status=pulp.LpSolveStatus.TimeLimit) if self._stage == 'cost' else stats

    monkeypatch.setattr(Optimizer, '_solve', cost_stopped)
    optimizer = build('026-attenuate-grid-peaks')
    optimizer.settings = OptimizerSettings(probe_seconds=0)
    result = optimizer.solve()

    assert optimizer.solve_path == 'split'
    assert result['status'] == 'Feasible', result['status']
