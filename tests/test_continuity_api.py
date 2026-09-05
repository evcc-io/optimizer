from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from test_continuity import build, seed_fragmented, starts

from optimizer.app import app


@pytest.mark.parametrize('second_c_min', [None, 0, 1000])
def test_api_returns_continuous_equal_price_sessions(second_c_min: float | None):
    model = build()
    if second_c_min is not None:
        battery = deepcopy(model.batteries[0])
        battery.c_min = second_c_min
        model.batteries.append(battery)
    request = {
        'batteries': [{key: value for key, value in asdict(battery).items() if value is not None} for battery in model.batteries],
        'time_series': asdict(model.time_series),
        'eta_c': 1,
        'eta_d': 1,
    }

    response = app.test_client().post('/optimize/charge-schedule', json=request)

    assert response.status_code == 200
    result = response.get_json()
    assert result['status'] == 'Optimal'
    assert result['objective_value'] == pytest.approx(-0.45 * len(model.batteries), abs=2e-5)
    for config, battery in zip(model.batteries, result['batteries']):
        if config.c_min > 0:
            assert starts(battery['charging_power']) == 1
        assert battery['state_of_charge'][-1] == pytest.approx(1500, abs=0.1)


@pytest.mark.parametrize('strategy', ['attenuate_demand_peaks', 'attenuate_feedin_peaks', 'attenuate_grid_peaks'])
def test_each_grid_peak_is_preserved(monkeypatch: pytest.MonkeyPatch, strategy: str):
    model = build(strategy)
    model.time_series.gt = [0, 0, 2000, 0, 0, 0]
    model.time_series.ft = [0, 0, 0, 0, 2000, 0]
    seed_fragmented(model, monkeypatch)
    with monkeypatch.context() as context:
        context.setattr('optimizer.optimizer.minimize_interruptions', lambda *args: None)
        original = model.solve()

    result = model.solve()

    for side, key in [('imp', 'grid_import'), ('exp', 'grid_export')]:
        if side in model.peak_sides:
            assert max(result[key]) <= max(original[key]) + 0.01
    assert starts(result['batteries'][0]['charging_power']) <= starts(original['batteries'][0]['charging_power'])
    assert np.isfinite(result['objective_value'])
