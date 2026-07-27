# Agent Rules for the evcc optimizer

This file provides guidance to AI coding agents working in this repository.

## Project Overview

- The optimizer computes cost optimal home energy schedules for evcc: battery charging and discharging, grid import and export, given price, solar, and demand forecasts.
- The core is a Mixed Integer Linear Program (MILP) built with PuLP and solved with CBC, exposed over an HTTP API (Flask plus flask-restx).
- A small Go CLI client (`cmd/`, generated `client/`) calls the API and renders the result as tables and charts.
- Python 3.13 managed with `uv`. Go 1.24.

## Essential Commands

All Python commands run through `uv`. Use `uv run python`, never bare `python`, it is not on PATH.

- `make test` runs the Python test suite (`uv run pytest`)
- `make lint` formats and lints (`autopep8` then `ruff check --fix`)
- `make run` starts the API locally on port 7050 (`uv run python -m optimizer.app`)
- `make build` regenerates the Go client from the OpenAPI spec (`go generate ./...`)
- `make install` syncs dependencies (`uv sync`)
- `make loadtest` runs Locust load tests against a local instance

## Architecture

- `src/optimizer/optimizer.py` builds and solves the MILP. `Optimizer` assembles the variables, objective, and constraints, then `solve()` returns the result dict.
- `src/optimizer/app.py` is the Flask API. It parses the request into the dataclasses (`GridConfig`, `BatteryConfig`, `TimeSeriesData`, `OptimizationStrategy`), runs the optimizer, and marshals the response. The optimization endpoint is `POST /optimize/charge-schedule`.
- `src/optimizer/settings.py` holds runtime settings (solver threads, time limit, MIP gap, strategy weight) via pydantic-settings with the `OPTIMIZER_` env prefix.
- `openapi.yaml` is the source of truth for the API contract.
- `cmd/client.go` is the Go CLI client. `client/client.gen.go` is generated from `openapi.yaml` and must not be edited by hand.
- `tests/` holds the test suite, `test_cases/*.json` hold data driven scenarios.

## Domain Knowledge

The model is a maximization problem. Read these before changing it:

- The objective is maximized. Costs and penalties enter as negative terms, so every penalty coefficient must be strictly positive to actually penalize.
- Energy is in Wh, power limits in W, prices per Wh. Time steps carry individual durations `dt` in seconds, so convert power to energy with `dt / 3600`.
- Penalty coefficients scale from a positive floor, `np.max([max_import_price, 0.1e-3])`. This keeps penalties positive even when market prices are zero or negative.
- Charging and discharging strategies are cost-neutral tie-breakers. They add tiny soft terms (coefficient around `min_import_price * 1e-6`) that only decide between economically equal solutions. They are intentionally excluded from `get_clean_objective_value()`, which recomputes the real economic value without strategy incentives or penalties.
- The objective is assembled in two parts, `cost_objective` (real money: grid cost, export revenue, battery value, demand rate, penalties) and `preference_objective` (the tie breakers, peak and ramp weights, priorities). Only the second is multiplied by `OPTIMIZER_STRATEGY_WEIGHT`. A new term goes into the part it belongs to, `tests/test_objective_split.py` asserts the split stays exhaustive.
- The solve stops on two gaps, whichever comes first: `OPTIMIZER_GAP_ABS` in currency units and `OPTIMIZER_GAP_REL` as a share of the objective. The absolute one governs small bills, the relative one large ones, where a cent is too fine to express and the search sits on a proven schedule burning the time limit. `gap_abs` is multiplied by `OBJECTIVE_SCALE` on the way to CBC because the model is scaled; `gap_rel` is a ratio and goes over untouched. Anything that disables one gap in a test has to disable both, otherwise the assert measures where the search stopped.
- `get_clean_objective_value()` measures battery value as `(s[T-1] - s[0]) * p_a`, but `s[0]` already includes the first time step's charging, so energy charged in the first step is not counted as a gain. The optimization objective itself uses the absolute final state of charge, `s[-1] * p_a`. Two solutions that are equal in the real objective can therefore report different clean values. Keep optional charging off the first time step when designing cost-neutrality scenarios.
- Grid limits are soft: exceeding `p_max_imp` or `p_max_exp` is penalized rather than forbidden, so an over constrained request reports the violation instead of returning infeasible.
- The assembled objective is multiplied by `OBJECTIVE_SCALE` before it goes to the solver. Prices per Wh put the raw coefficients close to CBC's absolute tolerances, where real improvements are discarded as numerical noise. Scaling does not change the argmax. Keep new objective terms in the same unit and let the scaling do its work, do not compensate for it in individual coefficients.

## Python Coding Standards

- Line length is 160 (`ruff` and `autopep8` are configured to match). Run `make lint` before committing.
- ruff rule sets in use: `F`, `I`, `E`, `W`, `PL`. Respect import ordering (`I`).
- Reuse the NumPy helpers already present in the model rather than hand rolling loops.
- Type hint new public functions and dataclass fields.

## API and Code Generation

- Change the API by editing `openapi.yaml` and the matching flask-restx models in `app.py` together, then run `make build` to regenerate the Go client.
- Never edit `client/client.gen.go` by hand. The generator config is `tools/cfg.yaml`, invoked through `tools/generate.go`.

## Testing

- The suite runs with `uv run pytest`.
- `test_cases/*.json` are loaded by `tests/test_app.py`. Each file holds a `request` and an optional `expected_response`, and the test asserts the optimizer status and, with `numpy.isclose`, the objective value. Add a scenario by dropping in a JSON file.
- A test case with `"strict": true` additionally compares grid import, grid export, and the battery schedules with `numpy.allclose`. Use it for features that only pick between cost neutral alternatives, where the objective value barely moves. Do not set it on scenarios with several equally optimal schedules, the comparison would be arbitrary.
- Prefer a data driven case over a dedicated test module, so a failure points at the scenario rather than at a feature specific script.
- Cover both the success and the limit violation paths.

## Writing Style

- No em dashes in comments, commit messages, or docs. Use periods, commas, or colons.
- Project name is `evcc`, always lowercase.
- Uppercase acronyms in prose: MILP, API, HTTP, JWT, SoC.
- Commit subjects follow conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `infra:`, then a short description with no trailing period. Reference the issue when relevant, for example `fix: use max() for penalty scaling floor (closes #75)`.

## Comment Style

- Prefer self documenting code. Comment the why, not the what.
- Default to no comment. Add one only for a non obvious constraint, invariant, or surprising behavior, and keep it to one or two lines.
- Do not reference the current task, PR, or issue in code comments. Git history covers that.

## Pull Request Descriptions

Structure PR descriptions in this order. No headlines. Be concise.

1. References first line: link the related issue or PR (`closes #71`, `fixes #123`, `pairs with evcc-io/evcc#456`). A PR should almost always reference an issue, only skip for trivial fixes.
2. Intro: one or a few sentences framing what the PR does and why. The full problem belongs in the linked issue.
3. Bullet list: the most significant changes or user facing implications, most significant first.
4. TODO section only if open points remain.

Avoid file paths, line numbers, or code listings copied from the diff. Include a code snippet only when it conveys a contract more clearly than prose. No testing checklists, no co-author footers, no generator footers.
