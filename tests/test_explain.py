import pytest

from optimizer import explain
from optimizer.app import app, settings


@pytest.fixture
def payload():
    # four quarter hours and one full hour, soc read at the end of each hour
    return {
        'req': {
            'batteries': [{
                's_min': 0, 's_max': 10000, 's_initial': 5000,
                'c_min': 0, 'c_max': 5000, 'd_max': 5000, 'p_a': 0.0002,
                'charge_from_grid': True,
            }],
            'time_series': {
                'dt': [900, 900, 900, 900, 3600],
                'gt': [250, 250, 250, 250, 1000],
                'ft': [0, 0, 0, 0, 2000],
                'p_N': [0.0003, 0.0003, 0.0001, 0.0001, 0.0002],
                'p_E': [0.0001] * 5,
            },
        },
        'res': {
            'status': 'Optimal',
            'objective_value': -1.5,
            'batteries': [{
                'charging_power': [0, 0, 1250, 1250, 0],
                'discharging_power': [250, 250, 0, 0, 0],
                'state_of_charge': [4750, 4500, 5750, 7000, 7000],
            }],
            'grid_import': [0, 0, 1500, 1500, 0],
            'grid_export': [0, 0, 0, 0, 1000],
        },
        'details': {
            'timestamp': ['2026-10-15T10:00:00+02:00', '2026-10-15T10:15:00+02:00', '2026-10-15T10:30:00+02:00',
                          '2026-10-15T10:45:00+02:00', '2026-10-15T11:00:00+02:00'],
            'batteryDetails': [{'name': 'bat', 'title': 'Home battery', 'type': 'battery', 'capacity': 10}],
        },
        'currency': 'EUR',
        'language': 'de',
    }


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(settings, 'azure_openai_endpoint', 'https://example.invalid/')
    monkeypatch.setattr(settings, 'azure_openai_key', 'test-key')


def test_hours_fold_quarter_hours_into_their_hour(payload):
    rows = explain.hours(payload['req'], payload['res'], payload['details']['timestamp'])

    assert [r['hour'] for r in rows] == ['2026-10-15 10:00', '2026-10-15 11:00']
    first, second = rows
    assert first['price'] == pytest.approx(0.0002)  # duration weighted mean of the quarter hours
    assert first['batteries'][0] == {'charge': 2500, 'discharge': 500, 'soc': 7000}
    assert first['grid_import'] == 3000
    assert second['pv'] == 2000 and second['grid_export'] == 1000


def test_hours_without_timestamps_count_from_the_start(payload):
    rows = explain.hours(payload['req'], payload['res'])
    assert [r['hour'] for r in rows] == ['+0h', '+1h']


def test_digest_names_batteries_and_hours(payload):
    text = explain.digest(payload)

    assert '0: Home battery (battery, 10 kWh): soc 5.0 kWh within 0.0..10.0 kWh' in text
    assert 'may charge from grid, no grid discharge, remaining energy valued 0.200 EUR/kWh' in text
    assert text.endswith('2026-10-15 11:00 | 0.200 | 0.100 | 2.0 | 1.0 | 0.0/1.0 | 0.0/0.0/7.0')


def test_explain_relays_the_digest_in_the_requested_language(payload, configured, monkeypatch, capsys):
    seen = {}

    def fake_chat(_settings, messages):
        seen['messages'] = messages
        return 'Because the price drops at 10:30.', {'total_tokens': 42}

    monkeypatch.setattr(explain, 'chat', fake_chat)
    response = app.test_client().post('/optimize/explain', json=payload)

    assert response.status_code == 200, response.json
    assert response.json == {'explanation': 'Because the price drops at 10:30.'}
    system, user = seen['messages']
    assert 'IETF tag de' in system['content']
    assert '2026-10-15 10:00 | 0.200' in user['content']
    assert '"tokens": 42' in capsys.readouterr().out


def test_explain_reports_model_errors(payload, configured, monkeypatch):
    def failing_chat(_settings, _messages):
        raise explain.ExplainError('model returned 429: slow down')

    monkeypatch.setattr(explain, 'chat', failing_chat)
    response = app.test_client().post('/optimize/explain', json=payload)

    assert response.status_code == 502
    assert response.json['message'] == 'model returned 429: slow down'


@pytest.mark.parametrize('drop', ['req', 'res'])
def test_explain_requires_request_and_response(payload, configured, drop):
    del payload[drop]
    response = app.test_client().post('/optimize/explain', json=payload)

    assert response.status_code == 400
    assert drop in response.json['details']


def test_explain_requires_the_plan_status(payload, configured):
    del payload['res']['status']
    response = app.test_client().post('/optimize/explain', json=payload)

    assert response.status_code == 400
    assert response.json['message'] == 'res.status is a required property'


def test_explain_unconfigured_is_503(payload):
    response = app.test_client().post('/optimize/explain', json=payload)

    assert response.status_code == 503
