import json

import pytest

from optimizer.app import app


@pytest.fixture
def payload():
    return {
        'batteries': [{
            's_min': 0, 's_max': 10000, 's_initial': 5000,
            'c_min': 0, 'c_max': 5000, 'd_max': 0, 'p_a': 0.0001,
        }],
        'time_series': {
            'dt': [3600, 3600], 'gt': [1000, 1000], 'ft': [0, 0],
            'p_N': [0.0003, 0.0003], 'p_E': [0.0001, 0.0001],
        },
    }


@pytest.mark.parametrize('body', ['', '{', 'private-invalid-body'])
def test_malformed_json_returns_400(body, capsys):
    response = app.test_client().post('/optimize/charge-schedule', data=body, content_type='application/json')

    assert response.status_code == 400
    assert response.json['message']
    logged = capsys.readouterr().out
    assert json.loads(logged)['bad_request'] == {
        'path': '/optimize/charge-schedule',
        'reason': 'Invalid request body',
        'fields': [],
        'validator': None,
    }
    assert 'private-invalid-body' not in logged


def test_missing_fields_are_logged(capsys):
    response = app.test_client().post('/optimize/charge-schedule', json={})

    assert response.status_code == 400
    assert set(response.json['details']) == {'batteries', 'time_series'}
    assert 'errors' not in response.json
    assert json.loads(capsys.readouterr().out)['bad_request'] == {
        'path': '/optimize/charge-schedule',
        'reason': 'Input payload validation failed',
        'fields': ['batteries', 'time_series'],
        'validator': 'required',
    }


@pytest.mark.parametrize('validator', ['enum', 'type'])
def test_validation_logs_omit_request_values(payload, validator, capsys):
    secret = 'private-request-value'
    if validator == 'enum':
        payload['strategy'] = {'charging_strategy': secret}
        field = 'strategy.charging_strategy'
    else:
        payload['batteries'][0]['c_min'] = secret
        field = 'batteries.0.c_min'

    response = app.test_client().post('/optimize/charge-schedule?token=private-query', json=payload,
                                      headers={'Authorization': 'Bearer private-token'})

    assert response.status_code == 400
    assert secret in json.dumps(response.json['details'])
    logged = capsys.readouterr().out
    assert 'private-' not in logged
    assert json.loads(logged)['bad_request'] == {
        'path': '/optimize/charge-schedule',
        'reason': 'Input payload validation failed',
        'fields': [field],
        'validator': validator,
    }


def test_length_mismatch_preserves_cause(payload, capsys):
    payload['time_series']['p_N'] = [0.0003]

    response = app.test_client().post('/optimize/charge-schedule', json=payload)

    assert response.status_code == 400
    assert response.json == {'message': 'All time series must have the same length'}
    assert json.loads(capsys.readouterr().out)['bad_request'] == {
        'path': '/optimize/charge-schedule',
        'reason': 'All time series must have the same length',
        'fields': [],
        'validator': None,
    }


def test_conversion_error_logs_omit_exception_values(payload, monkeypatch, capsys):
    def invalid_battery(**kwargs):
        raise ValueError('private-exception-value')

    monkeypatch.setattr('optimizer.app.BatteryConfig', invalid_battery)

    response = app.test_client().post('/optimize/charge-schedule', json=payload)

    assert response.status_code == 400
    logged = capsys.readouterr().out
    assert 'private-exception-value' not in logged
    assert json.loads(logged)['bad_request']['reason'] == 'Invalid data format'


def test_other_statuses_do_not_log_bad_requests(payload, monkeypatch, capsys):
    response = app.test_client().post('/optimize/charge-schedule', json=payload)
    assert response.status_code == 200
    assert 'bad_request' not in capsys.readouterr().out

    monkeypatch.setenv('JWT_TOKEN_SECRET', 'test-secret')
    response = app.test_client().post('/optimize/charge-schedule', json={})
    assert response.status_code == 401
    assert 'bad_request' not in capsys.readouterr().out
