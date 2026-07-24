
import json
import pathlib

from optimizer.app import app

# 021 solved without a charging strategy. Before the objective was scaled, the solver stopped
# here on a solution about 0.59% below the optimum, so this pins the value it has to reach.
CASE = pathlib.Path('test_cases/021-min-pv-use-case-with-weird-behavior.json')
EXPECTED_OBJECTIVE = 1.4327495658487992


def test_optimum_is_not_cut_off_by_solver_tolerances():
    request = json.loads(CASE.read_text())["request"]
    request["strategy"] = {"charging_strategy": "none"}

    response = app.test_client().post("/optimize/charge-schedule", json=request)

    assert response.status_code == 200, f"request returned with status {response.status_code}"
    assert response.json["status"] == "Optimal"
    # not an equality check: a better solution stays acceptable, a worse one does not
    assert response.json["objective_value"] >= EXPECTED_OBJECTIVE * (1 - 1e-6), \
        f"objective value: {response.json['objective_value']}, expected at least: {EXPECTED_OBJECTIVE}"
