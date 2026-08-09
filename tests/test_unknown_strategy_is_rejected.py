import json

from optimizer.app import app
from optimizer.optimizer import CHARGING_STRATEGIES, DISCHARGING_STRATEGIES


def test_unknown_strategy_is_rejected():
    # a misspelled strategy reaches no branch in the optimizer, so the request used to be solved
    # without any strategy and came back Optimal with a cost optimal schedule - nothing in the
    # response says the strategy was dropped (#132)
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
            "p_E": [0.1, 0.1],
        },
    }

    for strategy in ({"charging_strategy": "attenuate_fedin_peaks"},
                     {"discharging_strategy": "discharge_before_export"},
                     {"charging_strategy": "discharge_before_import"},
                     {"charging_strategy": "not_a_real_strategy"}):
        response = client.post("/optimize/charge-schedule", json={**request, "strategy": strategy})
        assert response.status_code == 400, f"{strategy} was accepted"
        assert "not one of" in json.dumps(response.json["details"]), response.json

    # the documented names still pass, in every combination, and so does an absent strategy
    assert client.post("/optimize/charge-schedule", json=request).status_code == 200
    for charging in CHARGING_STRATEGIES:
        for discharging in DISCHARGING_STRATEGIES:
            strategy = {"charging_strategy": charging, "discharging_strategy": discharging}
            response = client.post("/optimize/charge-schedule", json={**request, "strategy": strategy})
            assert response.status_code == 200, f"{strategy} was rejected: {response.json}"
