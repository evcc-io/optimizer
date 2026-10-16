"""Fixtures shared by the explain and feedback tests."""
import pytest

from optimizer.app import settings


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

