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
| | Level deviation | `prc_p_dev = penalty_base·1e-4`, per step weighted by `dt[t]/Σdt` | ×1e-4·dt[t]/Σdt | 2.5e-5 – 2.5e-2 | shape, not cost |
| | Charge-before-export | `min_import_price·2e-5·(T−t)` | ×2e-5·(T−t) | ~1e-3 – tens | shape, not cost |
| | Discharge-before-import | `min_import_price·5e-6·(T−t)` | ×5e-6·(T−t) | ~1e-4 – tens | shape, not cost |
| | Charging priority | `min_import_price·5e-5·(T−t)·c_priority`, on `c` and on `d` | ×5e-5·(T−t)·c_priority | ~1e-3 – tens | shape, not cost |

## What the table does not show

- The SOC penalty is charged per time step, so a violation lasting the whole horizon is paid for
  once per step while one confined to a single step is paid for once. `penalty_base` is sized
  against the single-step case, which is the weaker of the two.
- The two leveling terms are ordered per W, not per horizon. Peak leveling is charged once on a
  single value, level deviation is a time average, so neither grows with the horizon and the order
  between them holds at any sampling.
- Level deviation carries the smallest coefficients in the model, a decade below peak leveling and
  smaller again once the time average is spread over the steps. Level and deviation variables are
  therefore bounded by the power the side can physically carry: unbounded they relax into the same
  range as the big M of 1e6 and, on the largest stored case, cost the solver a factor of ten.
- Level deviation has a blind spot. The level is free, and on a side that mostly rests at zero it
  settles there; with `p_grid ≥ 0` the term is then `Σ p_grid[t]·dt[t]`, the energy through the
  side, which the energy balance already fixes. A plateau, a jagged profile and a single spike of
  the same energy score identically and only the peak separates them. A ramp leveling term at
  `penalty_base·1e-5` used to order them; it was dropped because it priced transitions only and
  cost around a third of the solve time of a leveling case. `test_peak_leveling` pins both halves.
- The tier 3 terms keyed off `min_import_price` invert sign when market prices go negative, the
  reason peak and level deviation leveling use `penalty_base` instead.

