"""BulkHiGHS hands HiGHS the same model as pulp's own HiGHS, column by column and row by row."""
import highspy
import numpy as np
import pulp
import pytest
from test_objective_split import build

from optimizer.highs import BulkHiGHS


def arrays(solver, case):
    optimizer = build(case)
    optimizer.create_model()
    problem = optimizer.problem
    problem.solverModel = highspy.Highs()
    solver.buildSolverModel(problem)
    lp = problem.solverModel.getLp()
    return {
        'cost': list(lp.col_cost_), 'col_lower': list(lp.col_lower_), 'col_upper': list(lp.col_upper_),
        'integrality': list(lp.integrality_), 'row_lower': list(lp.row_lower_), 'row_upper': list(lp.row_upper_),
        'start': list(lp.a_matrix_.start_), 'index': list(lp.a_matrix_.index_), 'value': list(lp.a_matrix_.value_),
        'sense': lp.sense_, 'num_col': lp.num_col_, 'num_row': lp.num_row_,
    }


@pytest.mark.parametrize('case', ['012-early-charging-not-perfect', '026-attenuate-grid-peaks',
                                  '028-attenuate-grid-peaks-long-horizon'])
def test_same_model_as_pulp(case):
    stock = arrays(pulp.HiGHS(msg=False), case)
    bulk = arrays(BulkHiGHS(msg=False), case)

    assert bulk.keys() == stock.keys()
    for key, value in stock.items():
        assert bulk[key] == value, key
    assert stock['num_col'] > 10 and stock['num_row'] > 10  # a real model, not an empty one
    assert np.count_nonzero(np.array(stock['integrality']) == highspy.HighsVarType.kInteger) > 0
