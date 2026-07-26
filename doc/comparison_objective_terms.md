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
| | Demand-rate charge | `grid.prc_p_exc_imp` | *(raw; `prc_p_exc_imp / horizon_hours` now also feeds `penalty_base`)* | unbounded, but tier 1 now tracks it | actual currency |
| **3 — tie-break nudges** | Peak leveling | `prc_p_peak = penalty_base·1e-3` | ×1e-3 | 0.16 – 300 | shape, not cost |
| | Charge-before-export | `min_import_price·2e-5·(T−t)` | ×2e-5·(T−t) | ~1e-3 – tens | shape, not cost |
| | Discharge-before-import | `min_import_price·5e-6·(T−t)` | ×5e-6·(T−t) | ~1e-4 – tens | shape, not cost |
| | Charging priority | `min_import_price·5e-5·(T−t)·c_priority` | ×5e-5·(T−t)·c_priority | ~1e-3 – tens | shape, not cost |
| | Ramp leveling | `prc_p_ramp = penalty_base·1e-5` | ×1e-5 | 1.6e-3 – 3.0 | shape, not cost |

