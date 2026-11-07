import pytest

from optimizer import explain
from optimizer.app import app, settings


def test_step_labels_without_timestamps_count_from_the_start(payload):
    assert explain.step_labels(payload['req']['time_series']['dt']) == ['+0:00', '+0:15', '+0:30', '+0:45', '+1:00', '+2:00']


def test_step_labels_are_relative_to_the_first_day():
    stamps = ['2026-10-15T23:45:00.123456789+02:00', '2026-10-16T00:00:00+02:00', '2026-10-17T10:00:00+02:00']
    assert explain.step_labels([900, 900, 3600], stamps) == ['today 23:45', 'tomorrow 00:00', 'Saturday 10:00', 'Saturday 11:00']


def test_digest_describes_batteries_and_lists_every_step(payload):
    text = explain.digest(payload)

    assert 'Now: today 10:00. The plan runs until today 12:00. Status Optimal.' in text
    assert 'Prices in ct/kWh' in text
    assert 'objective' not in text
    assert ('home battery "Home battery": capacity 10.0 kWh, now 50% (5.0 kWh), kept between 0% and 100%, charges up to 5.0 kW, '
            'discharges up to 5.0 kW, may charge from the grid, does not discharge into the grid') in text
    assert 'today 10:30 | 15 | 10.0 | 10.0 | 0.0 | 0.2 | 1.5/0.0 | 1.2/0.0/57' in text
    assert text.endswith('today 11:00 | 60 | 20.0 | 10.0 | 2.0 | 1.0 | 0.0/1.0 | 0.0/0.0/70')

    payload['details']['batteryDetails'] = []
    assert 'home battery "home battery 1": capacity 10.0 kWh' in explain.digest(payload)


def test_digest_spells_out_goals_demands_and_limits(payload):
    payload['req']['batteries'][0].update({
        'd_max': 0, 'c_min': 1400, 'c_active': True, 's_max': 8000,
        's_goal': [0, 0, 0, 8000, 0], 'p_demand': [350, 350, 0, 0, 1000],
    })
    payload['details']['batteryDetails'][0] = {'title': 'Carport (i3s)', 'type': 'vehicle', 'capacity': 10}
    payload['details']['demandDetails'] = [{'type': 'home', 'values': []}, {'type': 'heating', 'title': 'Heat pump', 'values': []}]
    payload['req']['grid'] = {'p_max_imp': 11000}
    payload['res']['limit_violations'] = {'grid_import_limit_exceeded': True, 'grid_export_limit_hit': False}
    payload['currency'] = 'GBP'
    text = explain.digest(payload)

    assert ('vehicle "Carport (i3s)": capacity 10.0 kWh, now 50% (5.0 kWh), kept between 0% and 80%, '
            'charges up to 5.0 kW, at least 1.4 kW or not at all, never discharges, may charge from the grid, charging right now, '
            'charging goal 80% by today 11:00, must charge today 10:00 to today 10:30, must charge today 11:00 to today 12:00') in text
    assert 'grid import limited to 11.0 kW.' in text
    assert 'the plan needs more grid import than the limit allows.' in text
    assert 'The household load includes heating "Heat pump".' in text
    assert 'Prices in p/kWh' in text
    assert text.endswith('today 11:00 | 60 | 20.0 | 10.0 | 2.0 | 1.0 | 0.0/1.0 | 0.0/0.0/70')


def test_digest_prices_without_subunit_in_the_currency(payload):
    payload['currency'] = 'HUF'
    assert 'today 10:00 | 15 | 0.300 | 0.100' in explain.digest(payload)


def test_explain_replaces_dashes(payload, monkeypatch):
    monkeypatch.setattr(explain, 'chat', lambda _settings, _messages: ('Sunny — cheap 10:00–13:00.', {}))
    assert explain.explain(payload, settings)[0] == 'Sunny, cheap 10:00-13:00.'


def test_explain_relays_the_digest_in_the_requested_language(payload, configured, monkeypatch, capsys):
    seen = {}

    def fake_chat(_settings, messages):
        seen['messages'] = messages
        return 'Because the price drops at 10:30.', {'total_tokens': 42}

    monkeypatch.setattr(explain, 'chat', fake_chat)
    response = app.test_client().post('/optimize/explain', json=payload)

    assert response.status_code == 200, response.json
    assert response.json['explanation'] == 'Because the price drops at 10:30.'
    system, user = seen['messages']
    assert system['content'] == explain.SYSTEM_PROMPT + '\nAnswer in the language with IETF tag de.'
    assert 'today 10:00 | 15 | 30.0' in user['content']
    assert '"tokens": 42' in capsys.readouterr().out


@pytest.mark.parametrize('language, expected', [(None, 'en'), ('de-AT', 'de-AT')])
def test_explain_asks_for_the_language(payload, configured, monkeypatch, language, expected):
    seen = {}

    def fake_chat(_settings, messages):
        seen['system'] = messages[0]['content']
        return 'ok', {}

    monkeypatch.setattr(explain, 'chat', fake_chat)
    if language:
        payload['language'] = language
    else:
        del payload['language']

    assert app.test_client().post('/optimize/explain', json=payload).status_code == 200
    assert seen['system'].endswith(f'\nAnswer in the language with IETF tag {expected}.')


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
