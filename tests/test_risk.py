import copy
import json
import pathlib

import numpy
import pytest

from optimizer.app import app

# the two stored risk cases, each against the same request without the risk field. The stored
# case pins the early schedule, these pin that it is the field that moves it there.
DEPARTURE = json.loads(pathlib.Path('test_cases/029-departure-risk-charges-early.json').read_text())['request']
BAND = json.loads(pathlib.Path('test_cases/030-forecast-band-charges-early.json').read_text())['request']


def solve(request):
    response = app.test_client().post('/optimize/charge-schedule', json=request)
    assert response.status_code == 200, response.json
    assert response.json['status'] == 'Optimal'
    return response.json


def charging(result):
    return numpy.array(result['batteries'][0]['charging_power'])


def test_departure_probability_moves_the_charge_ahead_of_the_departure():
    without = copy.deepcopy(DEPARTURE)
    del without['batteries'][0]['p_departure']

    # the cheapest hour is the last one, so without the risk the charge waits for it
    assert charging(solve(without)).argmax() == 3
    # a 30 % chance of leaving in the first hour outweighs the 0.1 the wait would save
    assert charging(solve(DEPARTURE)).argmax() == 0


def test_departure_value_is_the_expected_state_of_charge():
    result = solve(DEPARTURE)
    bat = DEPARTURE['batteries'][0]
    soc = numpy.array(result['batteries'][0]['state_of_charge'])
    weights = numpy.array(bat['p_departure'], float)
    weights[-1] += 1 - weights.sum()
    expected = ((soc - bat['s_initial']) * weights).sum() * bat['p_a'] \
        - (numpy.array(result['grid_import']) * numpy.array(DEPARTURE['time_series']['p_N'])).sum()
    assert numpy.isclose(result['objective_value'], expected)


def test_forecast_band_moves_the_charge_to_the_surer_surplus():
    without = copy.deepcopy(BAND)
    del without['time_series']['ft_err']

    # the first surplus pays a little more exported, so at the forecast the battery waits for the second
    assert charging(solve(without)).argmax() == 3
    # the second surplus may fall 1000 Wh short, which the battery would then draw from the grid
    assert charging(solve(BAND)).argmax() == 1


def test_forecast_band_cost_is_the_mean_over_the_band():
    result = solve(BAND)
    series = BAND['time_series']
    net = numpy.array(result['grid_import']) - numpy.array(result['grid_export'])
    value = 0
    for err in (-1, 1):
        shifted = net - err * numpy.array(series['ft_err'])
        value += 0.5 * (numpy.minimum(shifted, 0) * -numpy.array(series['p_E'])
                        - numpy.maximum(shifted, 0) * numpy.array(series['p_N'])).sum()
    value += result['batteries'][0]['state_of_charge'][-1] * BAND['batteries'][0]['p_a']
    assert numpy.isclose(result['objective_value'], value)


@pytest.mark.parametrize('p_departure', [[0.5, 0.6, 0, 0], [-0.1, 0.5, 0, 0]])
def test_departure_probabilities_are_validated(p_departure):
    request = copy.deepcopy(DEPARTURE)
    request['batteries'][0]['p_departure'] = p_departure
    response = app.test_client().post('/optimize/charge-schedule', json=request)
    assert response.status_code == 400
    assert 'p_departure' in response.json['message']


def test_short_ft_err_is_a_length_mismatch():
    request = copy.deepcopy(BAND)
    request['time_series']['ft_err'] = [0, 200]
    response = app.test_client().post('/optimize/charge-schedule', json=request)
    assert response.status_code == 400
    assert response.json['lengths']['ft_err'] == 2
