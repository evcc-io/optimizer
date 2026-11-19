import copy
import json
import pathlib

import pytest

from optimizer.app import app


def test_c_min_rest_with_charge_demand(monkeypatch):
    """A vehicle short of a full c_min slot fills the same with an inactive p_demand as without (#309)."""
    monkeypatch.setenv("OPTIMIZER_GAP_ABS", "0.01")
    client = app.test_client()
    request = json.loads(pathlib.Path('test_cases/030-c_min-short-first-slot.json').read_text())['request']

    with_demand = copy.deepcopy(request)
    with_demand['batteries'][0]['p_demand'] = [0.] * len(request['time_series']['dt'])

    plain = client.post('/optimize/charge-schedule', json=request).get_json()
    demand = client.post('/optimize/charge-schedule', json=with_demand).get_json()

    assert demand['objective_value'] == pytest.approx(plain['objective_value'], abs=0.01)
