
import json
import pathlib

import numpy
import pytest

from optimizer.app import app


@pytest.mark.parametrize('test_case', pathlib.Path('test_cases').glob('*.json'))
def test_optimizer(test_case: pathlib.Path):
    client = app.test_client()

    test_data = json.loads(test_case.read_text())

    request = test_data["request"]
    expected_response = test_data.get("expected_response")

    response = client.post("/optimize/charge-schedule", json=request)

    assert response.status_code == 200, f"request returned with status {response.status_code}"

    if expected_response is not None:
        # check optimizer status
        actual_status = response.json["status"]
        expected_status = expected_response.get("status", {})
        assert actual_status == expected_status, \
            f"optimizer status: {actual_status}, expected was: {expected_status}"
        # check objective value
        actual_objective_value = response.json["objective_value"]
        expected_objective_value = expected_response.get("objective_value", {})
        assert numpy.isclose(actual_objective_value,
                             expected_objective_value,
                             rtol=1e-05, atol=1e-08, equal_nan=False), \
            f"objective value: {actual_objective_value}, expected was: {expected_objective_value}"
    # cases marked strict also compare the schedule itself. needed where the feature under
    # test only picks between cost neutral alternatives, which the objective value hides
    if test_data.get("strict"):
        for key in ("grid_import", "grid_export"):
            assert numpy.allclose(response.json[key], expected_response[key], atol=1), \
                f"{key}: {response.json[key]}, expected was: {expected_response[key]}"
        for i, battery in enumerate(expected_response["batteries"]):
            for key in ("charging_power", "discharging_power"):
                actual = response.json["batteries"][i][key]
                assert numpy.allclose(actual, battery[key], atol=1), \
                    f"battery {i} {key}: {actual}, expected was: {battery[key]}"


def test_abort_returns_json_message():
    # message-only api.abort(400, ...) must return a JSON body, not an empty response
    client = app.test_client()
    request = {
        "batteries": [{
            "s_min": 0, "s_max": 10000, "s_initial": 5000,
            "c_min": 0, "c_max": 5000, "d_max": 5000, "p_a": 0.1,
        }],
        "time_series": {
            "dt": [3600, 3600],
            "gt": [1000, 1000],
            "ft": [0, 0],
            "p_N": [0.3, 0.3],
            "p_E": [0.1],  # length mismatch triggers api.abort(400, message)
        },
    }
    response = client.post("/optimize/charge-schedule", json=request)
    assert response.status_code == 400
    assert response.json is not None
    assert "message" in response.json
