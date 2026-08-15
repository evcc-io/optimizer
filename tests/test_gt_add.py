from optimizer.app import app


def request_with(time_series):
    return {
        "batteries": [{
            "s_min": 0, "s_max": 10000, "s_initial": 5000,
            "c_min": 0, "c_max": 5000, "d_max": 5000, "p_a": 0.1,
        }],
        "time_series": {"dt": [3600, 3600], "ft": [0, 0], "p_N": [0.3, 0.3], "p_E": [0.1, 0.1], **time_series},
    }


def test_additional_base_loads_are_summed_into_gt():
    client = app.test_client()

    split = client.post("/optimize/charge-schedule", json=request_with({"gt": [400, 500], "gt_add": [[100, 200], [500, 300]]}))
    summed = client.post("/optimize/charge-schedule", json=request_with({"gt": [1000, 1000]}))

    assert split.status_code == 200, split.json
    assert split.json == summed.json


def test_additional_base_load_of_different_length_is_rejected():
    client = app.test_client()

    response = client.post("/optimize/charge-schedule", json=request_with({"gt": [1000, 1000], "gt_add": [[100]]}))

    assert response.status_code == 400, response.json
