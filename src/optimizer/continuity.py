from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pulp

if TYPE_CHECKING:
    from .optimizer import Optimizer

TIME_LIMIT = 1.0
TOLERANCE = 1e-5


def minimize_interruptions(model: Optimizer, tmpdir: str, deadline: float | None) -> None:
    """Prefer fewer charge starts without trading away economics or existing preferences."""
    if model.problem.sol_status not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible) or not model._is_integral():
        return

    eligible = [i for i, active in model.variables['z_c'].items() if active is not None]

    def count_starts() -> list[int]:
        counts = []
        for i in eligible:
            active = np.array([pulp.value(v) for v in model.variables['c'][i]]) > TOLERANCE
            counts.append(int(np.count_nonzero(active & ~np.r_[False, active[:-1]])))
        return counts

    before = count_starts()
    if not any(count > 1 for count in before):
        return
    remaining = TIME_LIMIT if deadline is None else min(TIME_LIMIT, deadline - time.monotonic())
    if remaining <= 0:
        return

    solution = {var: var.varValue for var in model.problem.variables()}
    cost = pulp.value(model.cost_objective)
    preference = pulp.LpAffineExpression(model.preference_objective)
    # Normalize tiny preference coefficients so CBC's row tolerance cannot erase their bound.
    scale = 1 / max((abs(value) for value in preference.values() if value), default=1)
    preference *= scale
    preferred = pulp.value(preference)
    peaks = [model.variables[f'p_{side}_peak'] for side in model.peak_sides]
    peak_values = [pulp.value(peak) for peak in peaks]

    # A shallow copy shares solution variables but leaves the reusable model's constraints intact.
    candidate = model.problem.copy()
    candidate += model.cost_objective >= cost - TOLERANCE
    candidate += preference >= preferred - TOLERANCE
    for peak, value in zip(peaks, peak_values):
        candidate += peak <= value + TOLERANCE
    starts = []
    for i in eligible:
        active = model.variables['z_c'][i]
        for t in model.time_steps:
            start = pulp.LpVariable(f'charge_start_{i}_{t}', lowBound=0, upBound=1)
            candidate += start >= active[t] - (active[t - 1] if t else 0)
            starts.append(start)
    candidate.setObjective(-pulp.lpSum(starts))

    improved = False
    try:
        if deadline is not None:
            remaining = min(remaining, deadline - time.monotonic())
            if remaining <= 0:
                return
        candidate.solve(model._solver(tmpdir, timeLimit=remaining))
        improved = (candidate.sol_status in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible)
                    and candidate.valid(TOLERANCE)
                    and model._is_integral()
                    and pulp.value(model.cost_objective) >= cost - 2 * TOLERANCE
                    and pulp.value(preference) >= preferred - 2 * TOLERANCE
                    and all(pulp.value(peak) <= value + 2 * TOLERANCE for peak, value in zip(peaks, peak_values))
                    and sum(count_starts()) < sum(before))
    except pulp.PulpSolverError:
        return
    finally:
        if not improved:
            for var, value in solution.items():
                var.varValue = value
