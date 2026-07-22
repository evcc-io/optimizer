"""
Battery charging profiles by cell chemistry.

Each profile defines how the maximum charge power decreases as a function
of State of Charge (SoC). This models the real-world CC-CV charging behavior
where the Battery Management System (BMS) reduces the charge current at
higher SoC to protect cells and minimize resistive losses.

Usage:
    from charging_profiles import get_profile, max_charge_power

    profile = get_profile("lifepo4_conservative")
    watts = max_charge_power(profile, capacity_wh=31000, soc_percent=75)

To find values for your own system:
    - Set c_rate_max to the highest C-rate your battery sustains
      without the BMS reducing the charge current
    - Set knee to the SoC percentage where the BMS starts
      reducing the charge current below c_rate_max
    - Leave k and c_rate_float at their defaults unless you have
      your own measurements (fit with Wolfram Alpha:
      "exponential fit {SoC1,I1}, {SoC2,I2}, ...")

General advice: always stay slightly below your measured values.
If your BMS starts tapering at 66%, set knee to 65%. If your
inverter sustains 160A, set c_rate_max to match C/4 or lower.
This avoids triggering BMS intervention, reduces resistive losses,
and extends cell life.
"""

import math

# ---------------------------------------------------------------------------
# Chemistry profiles
# ---------------------------------------------------------------------------

CHEMISTRY_PROFILES = {
    "lifepo4_conservative": {
        # Charging profile for LiFePO4 cells optimized for longevity and
        # minimal resistive losses. Derived from real-world measurements
        # on a mixed environment of aged LiYFePO4 cells (200Ah + 90Ah)
        # and newer LiFePO4 cells (314Ah) with Victron inverters.
        #
        # The charging curve has two phases:
        #   1. Below 'knee': constant power at 'c_rate_max' rate
        #   2. Above 'knee': power drops exponentially toward 'c_rate_float'
        #
        # Formula above knee:
        #   P = max(c_rate_float, c_rate_max * e^(-k * (SoC - knee)))

        "c_rate_max": 0.25,
        # Maximum charge rate below knee, as fraction of capacity.
        # C/4 means a 31 kWh battery charges at up to 7750 W.

        "knee": 65,
        # SoC percentage where charge current begins to taper.

        "k": 0.052,
        # Exponential decay constant (R^2 = 0.996 from BMS measurements).

        "c_rate_float": 0.01,
        # Minimum charge rate (maintenance/balancing phase).
        # C/100 means a 31 kWh battery trickle-charges at 310 W.
    },

    "lifepo4_moderate": {
        # Moderate profile for newer LiFePO4 cells in good condition.

        "c_rate_max": 0.33,     # C/3
        "knee": 75,
        "k": 0.045,
        "c_rate_float": 0.02,   # C/50
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_profiles():
    """Return list of available profile names."""
    return list(CHEMISTRY_PROFILES.keys())


def get_profile(name, **overrides):
    """
    Get a charging profile by name with optional overrides.

    Args:
        name: Profile name (e.g. "lifepo4_conservative")
        **overrides: Override individual values (e.g. knee=70, c_rate_max=0.2)

    Returns:
        dict with keys: c_rate_max, knee, k, c_rate_float
    """
    if name not in CHEMISTRY_PROFILES:
        available = ", ".join(CHEMISTRY_PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")

    profile = dict(CHEMISTRY_PROFILES[name])
    for key, value in overrides.items():
        if key not in profile:
            raise ValueError(f"Unknown parameter '{key}'. Valid: {', '.join(profile.keys())}")
        profile[key] = value

    return profile


def max_charge_power(profile, capacity_wh, soc_percent):
    """
    Calculate maximum charge power in watts for a given SoC.

    Args:
        profile: Charging profile dict from get_profile()
        capacity_wh: Battery capacity in Wh
        soc_percent: Current state of charge (0-100)

    Returns:
        Maximum charge power in watts
    """
    c_rate_max = profile["c_rate_max"]
    knee = profile["knee"]
    k = profile["k"]
    c_rate_float = profile["c_rate_float"]

    if soc_percent <= knee:
        c_rate = c_rate_max
    else:
        c_rate = max(c_rate_float, c_rate_max * math.exp(-k * (soc_percent - knee)))

    return capacity_wh * c_rate
