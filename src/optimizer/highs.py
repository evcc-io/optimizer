"""pulp's HiGHS solver with the model built in bulk.

pulp 4.0.0 feeds HiGHS one addCol and one addRow call per column and row, and re-reads the
objective for every column. On this model (3k columns, 3.5k rows) that build costs 0.028 s per
solve natively, 6.5 % of a HiGHS request at the median and 15 % at p90 in the amd64 benchmark,
and every stage of a request pays it again. One addCols and one addRows call bring it to
0.004-0.006 s. This is coin-or/pulp#957 verbatim, kept here until a pulp release carries it;
dropping this module and the BulkHiGHS reference in Optimizer._solver is then the whole change.
"""
import warnings
from math import inf

import highspy
import numpy as np
import pulp
from pulp import constants


class BulkHiGHS(pulp.HiGHS):
    def buildSolverModel(self, lp):
        obj_mult = -1 if lp.sense == constants.LpMaximize else 1

        exported_vars = lp.exported_variables()
        id_to_col = {v.id: j for j, v in enumerate(exported_vars)}

        cost = np.zeros(len(exported_vars))
        for var, coefficient in lp.objective.items():
            j = id_to_col.get(var.id)
            if j is not None:
                cost[j] = obj_mult * coefficient
        lower = np.array([-inf if var.lowBound is None else var.lowBound for var in exported_vars], dtype=np.float64)
        upper = np.array([inf if var.upBound is None else var.upBound for var in exported_vars], dtype=np.float64)
        empty = np.zeros(0, dtype=np.int32)
        lp.solverModel.addCols(len(exported_vars), cost, lower, upper, 0, empty, empty, np.zeros(0))

        if self.mip:
            integer = np.fromiter((j for j, var in enumerate(exported_vars) if var.cat == constants.LpInteger),
                                  dtype=np.int32)
            if len(integer):
                lp.solverModel.changeColsIntegrality(len(integer), integer,
                                                     np.full(len(integer), highspy.HighsVarType.kInteger))

        constraints = lp.constraints()
        starts: list[int] = []
        indices: list[int] = []
        coefficients: list[float] = []
        row_lower: list[float] = []
        row_upper: list[float] = []
        for constraint in constraints:
            starts.append(len(indices))
            for var, coefficient in constraint.items():
                if coefficient != 0:
                    indices.append(id_to_col[var.id])
                    coefficients.append(coefficient)
            lb = constraint.getLb()
            ub = constraint.getUb()
            row_lower.append(-inf if lb is None else lb)
            row_upper.append(inf if ub is None else ub)
        lp.solverModel.addRows(len(constraints),
                               np.array(row_lower, dtype=np.float64), np.array(row_upper, dtype=np.float64),
                               len(indices), np.array(starts, dtype=np.int32), np.array(indices, dtype=np.int32),
                               np.array(coefficients, dtype=np.float64))

        if self.optionsDict.get("warmStart", False):
            indices = []
            values: list[float] = []
            for j, var in enumerate(exported_vars):
                if var.varValue is not None:
                    indices.append(j)
                    values.append(var.varValue)
            if not indices:
                warnings.warn("No variable with value found: warmStart aborted")
                return
            lp.solverModel.setSolution(len(indices), indices, values)
