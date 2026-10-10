# optimizer

[![Go Reference](https://pkg.go.dev/badge/github.com/evcc-io/optimizer.svg)](https://pkg.go.dev/github.com/evcc-io/optimizer)

An HTTP service that plans the cheapest way to run a home energy system over the next hours or days.

Given a PV forecast, the expected household demand, dynamic grid prices and a description of every battery in the house, it returns a per-time-step schedule: how much to import, how much to export, and how much to charge or discharge each battery. The problem is expressed as a Mixed Integer Linear Program and solved with CBC via [PuLP](https://github.com/coin-or/pulp).

Inspired by https://github.com/Akkudoktor-EOS/EOS/pull/462

## What it does

- **Maximizes economic benefit** over the horizon: grid import cost, export revenue, and the value of the energy left in the batteries at the end.
- **Handles many batteries at once** — home storage and EVs — each with its own capacity, power limits, priority and permission to charge from or discharge to the grid.
- **Respects charging goals**: an EV can be required to reach a given state of charge by a given time step, or to charge at a minimum power while it is plugged in.
- **Honours grid limits** for import and export power, and supports a demand rate charged on the highest power drawn beyond a threshold.
- **Never returns "infeasible" for a goal it cannot reach.** Goals, minimum charge demand and grid limits are soft constraints backed by penalties, so an over-constrained request still yields the best achievable schedule plus a flag telling you which limit was violated.
- **Optional strategies** break ties that cost nothing: charge before exporting, discharge before importing, or level grid peaks on the import side, the feed-in side, or both.

## Example

Two days in 15 minute steps: a 10 kWh home battery at 50 %, a 50 kWh EV at 20 % that must reach 80 % by 08:00 on a 3.7 kW single phase connection, a 4 kWp PV forecast, and a dynamic tariff between 18 ct at noon and 44 ct at 19:00. Grid power is the line, PV forecast and household consumption the inputs, the bars are what the optimizer schedules for each battery.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/example-plan-dark.png">
  <img alt="Charging plan: the EV charges through the night at 3.7 kW, PV surplus goes into the EV and then the home battery, the battery covers the evening peak" src="docs/img/example-plan-light.png">
</picture>

The EV takes its 30 kWh in the eight cheapest hours before the deadline, from 22:00 to 06:00 at full power, and stops exactly at the goal. The morning PV surplus then goes into the EV and after noon into the home battery rather than to the grid, because `charge_before_export` makes self-consumption the tie-breaker. From there the battery covers the 44 ct evening peak and the night, and the house imports nothing until the next PV surplus arrives.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/example-soc-dark.png">
  <img alt="SoC projections: the EV climbing from 20 to 80 percent overnight and to full on PV, the home battery cycling between its overnight low and full" src="docs/img/example-soc-light.png">
</picture>

## Levelling grid peaks

Cheapest is not always kindest to the grid connection. Above, the home battery fills in one block after noon and the rest of the midday surplus leaves as a 3 kW feed-in peak.

`attenuate_grid_peaks` penalizes the highest grid power over the horizon, on the import and the feed-in side. The same household then spreads the battery refill over the whole solar window and holds the feed-in to a flat 1 kW plateau. The night import stays where it is: the EV needs its eight hours at 3.7 kW either way. Nothing costs more, the connection just sees a calmer profile. `attenuate_demand_peaks` and `attenuate_feedin_peaks` do the same for one side only.

The maximum is a single value out of the horizon, which leaves one gap: a load spike the schedule cannot touch, an oven, a heat pump defrost, fixes it, and the penalty then has nothing left to win below it. Charging flat out against the spike scores the same as spreading the same energy over the window, and the solver may pick either.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/example-peak-dark.png">
  <img alt="Charging plan with attenuate_grid_peaks: the battery refill spread over the solar window and the feed-in held to a flat 1 kW plateau" src="docs/img/example-peak-light.png">
</picture>

## How a request is solved

One request is up to five solver runs under one wall clock, `OPTIMIZER_TIME_LIMIT` (10 s in production). Each stage keeps what the previous one found unless it can improve on it without spending money. The request log carries where the clock went (`stages`), which path was taken (`path`), what the tie break (`preferences`) and the continuity pass (`continuity`) did, and what the cost stage found (`cost_stage_value`) against what it could not rule out (`cost_stage_gap`, currency, zero when proven).

```mermaid
flowchart TD
    build["build<br/>MILP, objective scaled"] --> probe{"probe<br/>cost and preferences in one solve, PROBE_SHARE of the limit"}
    probe -- "proven optimal" --> joint["path joint"]
    probe -- "not proven" --> cost{"cost<br/>money only, stopped on OPTIMIZER_GAP_ABS, at most COST_TIME_LIMIT"}
    cost -- "usable schedule" --> lp["tie_break LP<br/>binaries pinned, slack ladder if CBC calls the bound infeasible"]
    cost -- "no usable schedule" --> kept["path split, kept the probe"]
    lp --> milp["tie_break MILP<br/>clock permitting, at most MILP_PREFERENCE_TIME_LIMIT"]
    gate{"continuity gate<br/>a battery has more than one session, and the probe or the cost stage took at most CONTINUITY_TIME_LIMIT"}
    joint --> gate
    milp --> gate
    kept --> gate
    gate -- "joint: cap CONTINUITY_TIME_LIMIT<br/>split: cap CONTINUITY_SPLIT_TIME_LIMIT" --> continuity["continuity<br/>fewest charge starts under the cost, preference and peak bounds"]
    gate -- "not needed, or skipped" --> result
    continuity --> result["result<br/>Optimal, Feasible or Not Solved"]
```

| Stage | Task | Runs when | Parameters |
|---|---|---|---|
| `build` | Build the MILP and scale its objective so the largest coefficient sits at `OBJECTIVE_TARGET`. | Always. | `OBJECTIVE_TARGET` 1e6 |
| `probe` | Solve cost and preferences together. Proven optimal means the tie is decided in one solve: path `joint`, nothing else runs on the money. | Unless `OPTIMIZER_PROBE_SECONDS` is 0. | `OPTIMIZER_PROBE_SECONDS`, default `PROBE_SHARE` 0.2 of the limit; the same absolute gap as the cost stage |
| `cost` | Money only, stopped on an absolute gap. Holds back a slice of the clock for the tie break instead of taking whatever is left. | The probe did not prove its answer: path `split`. | `OPTIMIZER_GAP_ABS` 0.01 currency, raised to `OPTIMIZER_GAP_SHARE` 0.25 % of the goal energy at the mean import price, `COST_TIME_LIMIT` 3 s, `PREFERENCE_TIME_SHARE` 0.25 of the limit reserved |
| `tie_break`, LP | Pin the binaries the cost stage chose and move only the continuous variables, under a bound that keeps the cost found. Milliseconds, so it runs whatever the clock says. If CBC calls the bound infeasible the slack is widened tenfold per retry. | Path `split` and a strategy is configured. | `OPTIMIZER_PREFERENCE_BUDGET` 0, `COST_BOUND_SLACK` 1e-5 up to `COST_BOUND_SLACK_CEILING` 1e-2, `COST_BOUND_TOLERANCE` 1e-4, `LP_PREFERENCE_TIME_LIMIT` 1 s |
| `tie_break`, MILP | Search the whole model under the same bound to beat the LP. Whichever is ahead is returned. | After the LP, clock permitting. | The reserved slice, capped at `MILP_PREFERENCE_TIME_LIMIT` 2.5 s; uncapped without a time limit |
| `continuity` | Fewest charge starts for batteries with `c_min > 0`, bounded by the cost, the preference value and each levelled grid peak already reached. A preference, not a guarantee: prices, charge demands and grid shaping still win, and power may vary within a session. A device reported as charging (`c_active`) enters the horizon switched on, so keeping it on costs no start. | A battery has more than one charging session, and the solve the candidate extends took at most `CONTINUITY_TIME_LIMIT`: the probe on path `joint`, the cost stage on path `split`. | `CONTINUITY_TIME_LIMIT` 1 s, also the cap on path `joint`; `CONTINUITY_SPLIT_TIME_LIMIT` 2.5 s, the cap on path `split`; both bounded by the deadline; `CONTINUITY_TOLERANCE` 1e-5 |

A schedule the solver stopped on at the limit is reported as `Feasible` rather than `Optimal`. A solve that comes back off the integers or off the model's rows is refused and reported as `Not Solved`, whatever status CBC gave it.

## API

`POST /optimize/charge-schedule` takes the whole problem as one JSON document and returns the schedule. `GET /optimize/health` is the liveness probe. Every field is documented in [`openapi.yaml`](openapi.yaml).

```jsonc
{
  "strategy": { "charging_strategy": "charge_before_export" },
  "grid": { "p_max_exp": 7000 },                                             // W
  "batteries": [
    {
      "s_capacity": 52000, "s_max": 50000, "s_min": 0, "s_initial": 12000,   // Wh
      "c_min": 1380, "c_max": 11000, "d_max": 0,                             // W
      "s_goal": [0, 0, 0, 0, 0, 0, 0, 40000, 0, 0, 0, 0],                    // Wh per step
      "charge_from_grid": true,
      "p_a": 0.00022                                     // value of stored energy, per Wh
    }
  ],
  "time_series": {
    "dt": [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600],
    "gt": [230, 210, 205, 205, 240, 380, 780, 920, 640, 480, 430, 460],   // demand, Wh
    "ft": [0, 0, 0, 0, 0, 60, 320, 850, 1600, 2500, 3300, 3900],          // PV forecast, Wh
    "p_N": [0.00026, 0.00024, 0.00023, 0.00023, 0.00025, 0.00030,
            0.00036, 0.00041, 0.00038, 0.00032, 0.00027, 0.00022],        // import, per Wh
    "p_E": [0.00008, 0.00008, 0.00008, 0.00008, 0.00008, 0.00008,
            0.00008, 0.00008, 0.00008, 0.00008, 0.00008, 0.00008]         // export, per Wh
  }
}
```

The response carries `status`, the `objective_value`, `grid_import` / `grid_export` per step, `charging_power` / `discharging_power` / `state_of_charge` per battery, and `limit_violations` together with the `grid_import_overshoot` and `grid_export_overshoot` series.

Time steps do not have to be equally long: `dt` is given per step, so a schedule can be fine grained for the next hour and coarse for tomorrow.

A small Go client for sending requests to a running service lives in [`cmd/client.go`](cmd/client.go); it prints the request, the resulting schedule and the objective value:

```sh
jq .request test_cases/024-attenuate-demand-peaks.json | go run ./cmd
```

## Development

Optimizer relies on `uv` and `make` being available.
Installation instructions for `uv` [can be found here](https://docs.astral.sh/uv/getting-started/installation/).

Once `uv` and `make` are available on the PATH, you can run `make run` to set up the project environment and run the optimizer service.

Linting and formatting is run with `make lint`.
The test suite is run with `make test`.
To add a new dependency to the project, run `uv add <dependency>`.
To upgrade all depdendencies to their latest version, run `make upgrade`.

To make sure that your contributions pass the CI pipeline, run `make lint` and `make test` before comitting or pushing your code.

If you are using VSCode, we recommend the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [autopep8](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8), and [ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extensions.
Set up `autopep8` as your formatter for Python files.
