"""Regression: charge power must be semi-continuous (0 or >= c_min) in every
slot, including charge-demand slots. Previously the min-charge constraint was
skipped whenever p_demand[t] > 0, letting the solver return sub-minimum power
(e.g. ~950 W for a charger whose real minimum is far higher)."""

from optimizer.app import app

C_MIN = 1400.0  # W, e.g. 1p x 230V x ~6A


def _request(p_demand):
    # 3 x 1h slots; no solar, charging only from priced grid, stored energy
    # worthless (p_a=0). Charging is therefore strictly costly, so the solver
    # charges the minimum needed to meet p_demand - which (pre-fix) is the
    # sub-c_min demand value itself.
    return {
        "batteries": [
            {
                "charge_from_grid": True,
                "s_min": 0.0,
                "s_max": 10000.0,
                "s_initial": 0.0,
                "p_demand": p_demand,
                "c_min": C_MIN,
                "c_max": 11000.0,
                "d_max": 0.0,
                "p_a": 0.0,
            }
        ],
        "time_series": {
            "dt": [3600.0, 3600.0, 3600.0],
            "gt": [0.0, 0.0, 0.0],
            "ft": [0.0, 0.0, 0.0],  # no solar -> charge only from priced grid
            "p_N": [0.30, 0.30, 0.30],  # expensive grid import
            "p_E": [0.0, 0.0, 0.0],  # export worthless
        },
    }


def _charge_power(response):
    dt = response.json  # marshalled result
    powers = dt["batteries"][0]["charging_power"]  # Wh per 1h slot == W
    return powers


def test_charge_demand_below_cmin_is_not_sub_minimum():
    # demand 500 Wh in slot 0 is BELOW c_min energy (1400 Wh) -> the trap
    client = app.test_client()
    resp = client.post("/optimize/charge-schedule", json=_request([500.0, 0.0, 0.0]))
    assert resp.status_code == 200

    powers = _charge_power(resp)
    for w in powers:
        assert w < 1.0 or w >= C_MIN - 1.0, f"sub-minimum charge power {w} W (c_min={C_MIN})"

    # non-vacuous: the demand must actually drive charging (>= c_min), otherwise
    # the assertion above passes trivially on an all-zero schedule.
    assert max(powers) >= C_MIN - 1.0, f"expected charging >= c_min, got {powers}"
