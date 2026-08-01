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


# ---------------------------------------------------------------------------
# Level algorithm
# ---------------------------------------------------------------------------


def find_optimal_level(
    profile,
    capacity_wh,
    soc_start,
    soc_target,
    solar_w,
    consumption_w=800,
    dt_seconds=900,
):
    """
    Find the optimal flat export level that charges the battery to
    target SoC while minimizing the peak grid export.

    Instead of sliding a fixed charge block, this algorithm finds the
    horizontal line (export ceiling) such that charging with the surplus
    above this line fills the battery exactly to the target SoC.

    In each slot, the charge power is:
        min(PV - consumption - level, profile_max(SoC))

    This allows variable charge power per slot: slow charging at the
    edges of the solar bell curve, faster in the middle -- but always
    limited by the SoC-dependent profile.

    The result is a flatter export curve compared to the fixed-block
    approach, because the battery absorbs more during peak hours
    (when SoC is still low) instead of ramping up early.

    Args:
        profile: Charging profile dict (c_rate_max, knee, k, c_rate_float)
        capacity_wh: Battery capacity in Wh
        soc_start: Starting SoC (0-100)
        soc_target: Target SoC (0-100)
        solar_w: List of solar power values per slot (watts)
        consumption_w: Household consumption in watts (default: 800)
        dt_seconds: Slot duration in seconds (default: 900)

    Returns:
        dict with schedule and metadata, or None if target unreachable.
        Keys: n_slots, peak_export_w, peak_without_charge_w,
              total_charged_kwh, export_level_w, schedule
        schedule entries: {solar_w, charge_w, export_w, soc}
    """
    if soc_start >= soc_target:
        return None

    n_solar = len(solar_w)
    if n_solar == 0:
        return None

    dt_h = dt_seconds / 3600.0
    energy_needed_wh = capacity_wh * (soc_target - soc_start) / 100

    # Peak export without any charging
    peak_without = max(max(0, s - consumption_w) for s in solar_w)

    # Find the target peak that minimizes the actual peak export.
    # The relationship between target peak and actual peak is not
    # monotonic: too aggressive (low target) fills the battery early,
    # leaving peak solar slots unprotected. Too conservative (high
    # target) doesn't charge enough. We sweep through candidates
    # and simulate each one to find the true minimum.
    n_steps = 200
    best_target = peak_without
    best_actual_peak = peak_without

    for step in range(n_steps + 1):
        candidate = peak_without * step / n_steps

        # Simulate charging with this target
        soc = soc_start
        charged_wh = 0
        actual_peak = 0
        for i in range(n_solar):
            surplus = max(0, solar_w[i] - consumption_w)
            available = max(0, surplus - candidate)
            p_max = max_charge_power(profile, capacity_wh, soc)
            charge_w = min(available, p_max)
            energy_wh = charge_w * dt_h
            charged_wh += energy_wh
            soc += energy_wh / capacity_wh * 100
            soc = min(soc, soc_target)
            export_w = max(0, surplus - charge_w)
            actual_peak = max(actual_peak, export_w)

        # Only consider if enough energy is charged
        if charged_wh >= energy_needed_wh * 0.99:
            if actual_peak < best_actual_peak:
                best_actual_peak = actual_peak
                best_target = candidate

    target_peak = best_target

    # Final simulation at the found target peak
    soc = soc_start
    total_charged_wh = 0
    schedule = []
    peak_export = 0

    for i in range(n_solar):
        surplus = max(0, solar_w[i] - consumption_w)
        available = max(0, surplus - target_peak)
        p_max = max_charge_power(profile, capacity_wh, soc)
        charge_w = min(available, p_max)
        remaining_wh = capacity_wh * (soc_target - soc) / 100
        if charge_w * dt_h > remaining_wh:
            charge_w = remaining_wh / dt_h
        energy_wh = charge_w * dt_h
        total_charged_wh += energy_wh
        soc += energy_wh / capacity_wh * 100
        soc = min(soc, soc_target)

        export_w = max(0, surplus - charge_w)
        peak_export = max(peak_export, export_w)

        schedule.append({
            "solar_w": solar_w[i],
            "charge_w": charge_w,
            "export_w": export_w,
            "soc": soc,
        })

    n_charge_slots = sum(1 for s in schedule if s["charge_w"] > 0)
    if n_charge_slots == 0:
        return None

    return {
        "n_slots": n_charge_slots,
        "peak_export_w": peak_export,
        "peak_without_charge_w": peak_without,
        "total_charged_kwh": total_charged_wh / 1000,
        "export_level_w": target_peak,
        "schedule": schedule,
    }
