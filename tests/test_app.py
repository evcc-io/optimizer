
import json
import pathlib

import jwt
import numpy
import pytest

from optimizer.app import app, settings


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
        # atol is a hundredth of a cent. The tie break stage gives away that much of the cost
        # optimum to stay feasible against the solver's own rounding, see COST_BOUND_SLACK and
        # COST_BOUND_TOLERANCE, and a case whose value is small in currency, 023 reports 0.58,
        # would otherwise flap on a difference three orders below anything economically meaningful.
        assert numpy.isclose(actual_objective_value,
                             expected_objective_value,
                             rtol=1e-05, atol=1e-04, equal_nan=False), \
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

    # independent invariant, not a comparison against stored output: with c_min set every step
    # charges either nothing or at least c_min, except when the battery is essentially full and
    # only a sub-c_min top-off remains. guards against regressions like #19, where p_demand let
    # charging slip below c_min far from s_max.
    if response.json["status"] == "Optimal":
        dt = request["time_series"]["dt"]
        for i, battery in enumerate(request["batteries"]):
            c_min = battery.get("c_min", 0)
            if not c_min:
                continue
            charging = response.json["batteries"][i]["charging_power"]
            soc = response.json["batteries"][i]["state_of_charge"]
            for t, charge in enumerate(charging):
                step = c_min * dt[t] / 3600
                room = battery["s_max"] - soc[t]
                assert charge <= 1 or charge >= step - 1 or room <= step + 1, \
                    f"battery {i} t={t}: charges {charge} below c_min step {step} with {room} Wh room to s_max"


def test_subject_logging_is_configurable(capsys, monkeypatch):
    monkeypatch.setenv("JWT_TOKEN_SECRET", "test-secret")
    token = jwt.encode({"sub": "someone"}, "test-secret", algorithm="HS256")
    client = app.test_client()
    headers = {"Authorization": f"Bearer {token}"}

    monkeypatch.setattr(settings, "log_subject", False)
    client.post("/optimize/charge-schedule", json={}, headers=headers)
    assert "subject:" not in capsys.readouterr().out

    monkeypatch.setattr(settings, "log_subject", True)
    client.post("/optimize/charge-schedule", json={}, headers=headers)
    assert "subject: someone" in capsys.readouterr().out


def test_slow_requests_are_dumped(tmp_path, monkeypatch):
    request = json.loads(pathlib.Path('test_cases/009-discharge-before-import.json').read_text())["request"]
    dump = tmp_path / "nested" / "slow.jsonl"
    client = app.test_client()
    # a limit of zero makes every solve count as exhausting it
    monkeypatch.setattr(settings, "time_limit", 0)

    monkeypatch.setattr(settings, "dump_slow_requests", None)
    client.post("/optimize/charge-schedule", json=request)
    assert not dump.exists()

    monkeypatch.setattr(settings, "dump_slow_requests", str(dump))
    client.post("/optimize/charge-schedule", json=request)
    client.post("/optimize/charge-schedule", json=request)
    lines = dump.read_text().splitlines()
    assert len(lines) == 2, "every slow request appends one line"
    assert json.loads(lines[0])["request"] == request
    assert json.loads(lines[1])["elapsed"] > 0


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
