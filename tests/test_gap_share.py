import numpy
import pytest
from test_objective_split import build


def test_the_gap_grows_with_the_goal_energy():
    # a fixed cent sits below the relaxation slack of a vehicle with c_min and a charge goal (#186),
    # so the gap follows the money the request moves and keeps the cent as its floor
    model = build('024-attenuate-demand-peaks')
    need = sum(max(max(bat.s_goal) - bat.s_initial, 0.) / model.eta_c for bat in model.batteries if bat.s_goal)
    assert need > 0, 'the case needs a charge goal'
    scaled = model.settings.gap_share * need * numpy.mean(model.time_series.p_N)

    assert model._gap_abs() == pytest.approx(max(model.settings.gap_abs, scaled))

    model.settings.gap_share = 0
    assert model._gap_abs() == model.settings.gap_abs

    model.settings.gap_abs = None
    assert model._gap_abs() is None


def test_a_request_without_a_goal_keeps_the_floor():
    model = build('012-early-charging-not-perfect')
    assert not any(bat.s_goal and max(bat.s_goal) > bat.s_initial for bat in model.batteries)
    assert model._gap_abs() == model.settings.gap_abs
