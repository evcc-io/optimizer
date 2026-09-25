import pytest

from optimizer.app import app

C_MAX, P_MAX = 11000, 11000


@pytest.fixture
def payload():
    battery = {'s_min': 0, 's_max': 20000, 's_initial': 0, 'c_min': 0, 'c_max': C_MAX, 'd_max': 0, 'p_a': 0.0003, 'charge_from_grid': True}
    return {
        'batteries': [dict(battery), dict(battery)],
        'time_series': {'dt': [3600] * 3, 'gt': [0] * 3, 'ft': [0] * 3, 'p_N': [0.0001, 0.0002, 0.0002], 'p_E': [0] * 3},
    }


def charge_sums(response):
    return [sum(step) for step in zip(*(bat['charging_power'] for bat in response.json['batteries']))]


def test_circuit_limits_charge_sum(payload):
    # the response rounds the schedule, so the sum may exceed the limit by a fraction of a Wh
    client = app.test_client()

    control = client.post('/optimize/charge-schedule', json=payload)
    assert control.status_code == 200
    assert max(charge_sums(control)) > P_MAX

    payload['circuits'] = [{'p_max': P_MAX, 'batteries': [0, 1]}]
    limited = client.post('/optimize/charge-schedule', json=payload)
    assert limited.status_code == 200
    assert limited.json['status'] == 'Optimal'
    assert all(total <= P_MAX + 1 for total in charge_sums(limited))


def test_non_binding_circuit_keeps_schedule(payload):
    client = app.test_client()
    control = client.post('/optimize/charge-schedule', json=payload).json

    payload['circuits'] = [{'p_max': 2 * C_MAX, 'batteries': [0, 1]}]
    assert client.post('/optimize/charge-schedule', json=payload).json == control


@pytest.mark.parametrize('circuit, reason', [
    ({'p_max': P_MAX, 'batteries': [0, 2]}, 'Circuit battery index out of range'),
    ({'p_max': P_MAX, 'batteries': [-1]}, 'circuits.0.batteries.0'),
    ({'p_max': P_MAX, 'batteries': [0, 0]}, 'circuits.0.batteries'),
    ({'p_max': P_MAX, 'batteries': []}, 'circuits.0.batteries'),
    ({'p_max': 0, 'batteries': [0]}, 'circuits.0.p_max'),
    ({'p_max': -1, 'batteries': [0]}, 'circuits.0.p_max'),
])
def test_invalid_circuit_is_rejected(payload, circuit, reason):
    payload['circuits'] = [circuit]
    response = app.test_client().post('/optimize/charge-schedule', json=payload)
    assert response.status_code == 400
    assert reason in [response.json['message'], *response.json.get('details', {})]
