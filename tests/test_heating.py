import numpy as np

from optimizer.app import app
from optimizer.optimizer import GridConfig, HeatingConfig, OptimizationStrategy, Optimizer, TimeSeriesData, fit_heating_model

# ground-truth thermal parameters used to synthesize history (per 15min slot)
ALPHA, BETA, GAMMA = -0.02, 0.004, 0.3


def synth_history(n=96, seed=1):
    """Simulate a leaky thermal store with known parameters."""
    rng = np.random.default_rng(seed)
    energies = list(rng.uniform(0, 500, n))
    temps = [45.0]
    for e in energies:
        t = temps[-1]
        temps.append(t + ALPHA * t + BETA * e + GAMMA)
    return temps, energies


def test_fit_recovers_parameters():
    temps, energies = synth_history()
    alpha, beta, gamma = fit_heating_model(temps, energies)
    assert np.isclose(alpha, ALPHA, atol=1e-8)
    assert np.isclose(beta, BETA, atol=1e-8)
    assert np.isclose(gamma, GAMMA, atol=1e-6)


def test_fit_rejects_short_history():
    try:
        fit_heating_model([45.0, 44.0], [100.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_heating_shifts_to_cheap_slots():
    # 8 slots of 15min: cheap first half, expensive second half.
    # Optimal plan banks heat while cheap and coasts through expensive slots.
    T = 8
    ts = TimeSeriesData(
        dt=[900] * T,
        gt=[0.0] * T,
        ft=[0.0] * T,
        p_N=[0.1e-3] * 4 + [1e-3] * 4,
        p_E=[0.0] * T,
    )
    heat = HeatingConfig(temp_min=54.0, temp_max=60.0, temp_initial=55.0,
                         c_max=3000.0, alpha=ALPHA, beta=BETA, gamma=GAMMA)
    opt = Optimizer(
        strategy=OptimizationStrategy('none', 'none'),
        grid=GridConfig(None, None, None),
        batteries=[],
        time_series=ts,
        heating=[heat],
    )
    result = opt.solve()

    assert result['status'] == 'Optimal'
    temperature = result['heating'][0]['temperature']
    energy = result['heating'][0]['heating_energy']

    # comfort band held (soft constraint, allow LP tolerance)
    assert min(temperature) >= heat.temp_min - 1e-3
    assert max(temperature) <= heat.temp_max + 1e-3
    # all heating happens in the cheap half
    assert sum(energy[:4]) > 100.0
    assert sum(energy[4:]) < 1e-3
    # heating energy shows up as grid import
    assert np.isclose(sum(result['grid_import']), sum(energy), atol=1e-6)


def test_api_heating_roundtrip():
    temps, energies = synth_history()
    T = 6
    request = {
        'batteries': [],
        'heating': [{
            'temp_min': 50.0,
            'temp_max': 60.0,
            'temp_initial': 52.0,
            'c_max': 3000.0,
            'history_temp': temps,
            'history_energy': energies,
        }],
        'time_series': {
            'dt': [900] * T,
            'gt': [100.0] * T,
            'ft': [0.0] * T,
            'p_N': [0.3e-3] * T,
            'p_E': [0.1e-3] * T,
        },
    }

    response = app.test_client().post('/optimize/charge-schedule', json=request)
    assert response.status_code == 200, response.text
    body = response.json
    assert body['status'] == 'Optimal'
    assert len(body['heating']) == 1
    assert len(body['heating'][0]['heating_energy']) == T
    assert len(body['heating'][0]['temperature']) == T
    assert min(body['heating'][0]['temperature']) >= 50.0 - 1e-3
