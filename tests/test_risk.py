import copy
import json
import pathlib

import numpy
import pytest

from optimizer.app import app

# the stored case pins the early schedule, these pin that it is the field that moves it there
DEPARTURE = json.loads(pathlib.Path('test_cases/029-departure-risk-charges-early.json').read_text())['request']


def solve(request):
    response = app.test_client().post('/optimize/charge-schedule', json=request)
    assert response.status_code == 200, response.json
    assert response.json['status'] == 'Optimal'
    return response.json


def charging(result):
    return numpy.array(result['batteries'][0]['charging_power'])


def test_departure_probability_moves_the_charge_ahead_of_the_departure():
    without = copy.deepcopy(DEPARTURE)
    del without['batteries'][0]['r_departure']

    # the cheapest hour is the last one, so without the risk the charge waits for it
    assert charging(solve(without)).argmax() == 3
    # a 30 % chance of leaving in the first hour outweighs the 0.1 the wait would save
    assert charging(solve(DEPARTURE)).argmax() == 0


def test_departure_value_is_the_expected_state_of_charge():
    result = solve(DEPARTURE)
    bat = DEPARTURE['batteries'][0]
    soc = numpy.array(result['batteries'][0]['state_of_charge'])
    weights = numpy.array(bat['r_departure'], float)
    weights[-1] += 1 - weights.sum()
    expected = ((soc - bat['s_initial']) * weights).sum() * bat['p_a'] \
        - (numpy.array(result['grid_import']) * numpy.array(DEPARTURE['time_series']['p_N'])).sum()
    assert numpy.isclose(result['objective_value'], expected)


def test_a_device_that_has_certainly_left_does_not_charge():
    # free surplus after the departure and a strategy that prefers charging over exporting it: the
    # value is nil either way, and only the bound keeps the tie break from charging a gone vehicle
    request = {
        'strategy': {'charging_strategy': 'charge_before_export'},
        'batteries': [{'charge_from_grid': True, 's_min': 0, 's_max': 20000, 's_initial': 0, 'c_min': 0, 'c_max': 5000,
                       'd_max': 0, 'p_a': 0.00035, 'r_departure': [0, 0, 1, 0, 0, 0]}],
        'time_series': {'dt': [3600] * 6, 'gt': [0] * 6, 'ft': [0, 0, 0, 3000, 3000, 0],
                        'p_N': [0.0003] * 6, 'p_E': [0] * 6},
    }

    result = solve(request)

    assert charging(result)[:3].tolist() == [5000, 5000, 5000]
    assert charging(result)[3:].tolist() == [0, 0, 0]
    assert result['grid_export'][3:5] == [3000, 3000]


@pytest.mark.parametrize('r_departure', [[0.5, 0.6, 0, 0], [-0.1, 0.5, 0, 0]])
def test_departure_probabilities_are_validated(r_departure):
    request = copy.deepcopy(DEPARTURE)
    request['batteries'][0]['r_departure'] = r_departure
    response = app.test_client().post('/optimize/charge-schedule', json=request)
    assert response.status_code == 400
    assert 'r_departure' in response.json['message']


def test_short_r_departure_is_a_length_mismatch():
    request = copy.deepcopy(DEPARTURE)
    request['batteries'][0]['r_departure'] = [0.3, 0]
    response = app.test_client().post('/optimize/charge-schedule', json=request)
    assert response.status_code == 400
    assert response.json['lengths']['r_departure'] == [2]
