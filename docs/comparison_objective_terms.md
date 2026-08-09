# Objective term hierarchy 

## Tier table

| Tier | Term | Formula | Multiple of `penalty_base` | Scaled coefficient range | Intent |
|---|---|---|---|---|---|
| **1 — hard-avoidance** | SOC limit exceedance | `prc_soc_exc_pen = penalty_base·1000` | ×1000 | 1.6e5 – 3.0e8 | must basically never happen |
| | Unmet charge goal | `prc_e_goal_pen = penalty_base·100` | ×100 | 1.6e4 – 3.0e7 | strong avoid |
| | Unmet charge demand | `prc_p_goal_pen = penalty_base·100`  | ×100 | 1.6e4 – 3.0e7 | strong avoid |
| | Import limit exceedance | `prc_e_grid_imp_pen = penalty_base·100` | ×100 | 1.6e4 – 3.0e7 | strong avoid |
| | Export limit exceedance | `prc_e_grid_exp_pen·(1−t·1e-5) = penalty_base·100·(≈1)` | ×100 | 1.6e4 – 3.0e7 | strong avoid |
| **2 — real economics** | Grid import cost | `p_N[t]` | *(raw, and now also feeds `penalty_base` via `max_import_price`)* | ~-1 – 3e5 | actual currency |
| | Grid export revenue | `p_E[t]` | *(raw, now also feeds `penalty_base` via `max_export_price`)* | ~0 – 3e5 | actual currency |
| | Final SOC value | `bat.p_a` | *(raw, now also feeds `penalty_base` directly)* | ~1.7e2 (up to whatever `p_a` is configured) | actual currency |
| | Demand-rate charge | `grid.prc_p_exc_imp` | *(raw; `prc_p_exc_imp·3600/min(dt)` now also feeds `penalty_base`, only when `p_max_imp` makes the rate active)* | unbounded, but tier 1 now tracks it | actual currency |
| **3 — tie-break nudges** | Peak leveling | `prc_p_peak = penalty_base·1e-3` | ×1e-3 | 0.16 – 300 | shape, not cost |
| | Charge-before-export | `min_import_price·2e-5·(T−t)` | ×2e-5·(T−t) | ~1e-3 – tens | shape, not cost |
| | Discharge-before-import | `min_import_price·5e-6·(T−t)` | ×5e-6·(T−t) | ~1e-4 – tens | shape, not cost |
| | Charging priority | `min_import_price·5e-5·(T−t)·c_priority`, on `c` and on `d` | ×5e-5·(T−t)·c_priority | ~1e-3 – tens | shape, not cost |

## What the table does not show

- The SOC penalty is charged per time step, so a violation lasting the whole horizon is paid for
  once per step while one confined to a single step is paid for once. `penalty_base` is sized
  against the single-step case, which is the weaker of the two.
- Peak leveling is charged once on a single value, not per step, so its weight does not grow with
  the horizon or change with the sampling.
- Peak leveling has a blind spot. The maximum is one value out of the horizon, so once a load the
  schedule cannot touch pins it, nothing orders the steps below it: a plateau, a jagged profile and
  a single spike of the same energy score identically. A ramp leveling term at `penalty_base·1e-5`
  used to order them; it was dropped because it priced transitions only, so it could not tell a
  straight climb from a plateau either, and the second absolute value system over the same grid
  power cost around a fifth of the solve time of a leveling case. `test_peak_leveling` pins both
  halves.
- The tier 3 terms keyed off `min_import_price` invert sign when market prices go negative, the
  reason peak leveling uses `penalty_base` instead.
- Tier 3 is not solved together with the tiers above it. `solve()` maximizes tiers 1 and 2 first,
  then maximizes tier 3 over the schedules that keep that value, so the distance between the tiers
  no longer decides whether a preference is respected. The ranges listed above are what the second
  stage works on, and it normalizes them off its own largest coefficient before solving.

