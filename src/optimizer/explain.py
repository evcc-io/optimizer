"""Plan explanations: the plan rendered as text and handed to a chat model."""
import json
import re
import urllib.error
import urllib.request
from datetime import datetime

# the prompt is not a request parameter: the service would be an open relay to the model otherwise
SYSTEM_PROMPT = """You explain the schedule of the evcc home energy optimizer to the owner of the home.
The plan comes as text in this shape: a header line with the solver status, the objective value, the number of
steps, the length of the horizon and the charging strategy; one line per vehicle and home battery with its kind,
name, capacity, current soc and soc bounds, charge and discharge power limits, whether it may charge from or
discharge to the grid, the value of its remaining energy, and whether it has charge demands or soc goals; then a
table with one row per step: step start time, step length in minutes, import and feed-in price per kWh, the solar
forecast and the household load forecast as energy for the step, the planned grid import/export, and per vehicle
and battery the planned charge/discharge energy and the soc at the end of the step. Steps get longer towards the
end of the horizon, the first step covers only the rest of the current one.
The optimizer solves a mixed integer linear program over the horizon. It minimises grid import cost minus
feed-in revenue minus the value of the energy left in each battery at the end of the horizon (p_a per kWh),
subject to the power balance, the battery bounds and efficiencies, grid limits, charge demands and soc goals.
A charging strategy only chooses between schedules of equal cost: charge_before_export charges batteries before
exporting to the grid, attenuate_demand_peaks levels the grid import profile by charging at partial power over
several steps instead of one peak, attenuate_feedin_peaks levels the grid export profile by charging to shave solar
feed-in peaks, attenuate_grid_peaks levels both. The discharging strategy discharge_before_import discharges
batteries before importing from the grid. A battery marked "no discharge" only ever charges, a vehicle for instance.
Status Optimal is the proven cost optimum,
Feasible is a schedule the solver stopped on at its time limit, Infeasible means the constraints contradict
each other and no schedule exists.
Start the answer with one sentence on what the plan covers: the horizon, the number of steps and the vehicles and
batteries in it. Then explain why the plan looks the way it does: when and why each vehicle and home battery
charges or discharges, how prices, solar and load drive that, and what state it ends in. Call every vehicle and
home battery by the name given in the plan, say vehicle for a vehicle and battery for a home battery. Point out
anything surprising. Be concrete, cite times and numbers from the plan. Use markdown with short paragraphs, no
tables. Stay under 250 words."""

CHAT_TIMEOUT = 45  # seconds, below the gunicorn request timeout


class ExplainError(Exception):
    """The model could not be asked."""


def at(series, i, default=0):
    return series[i] if series and i < len(series) else default


def step_labels(dt, timestamps=None):
    """One label per step: the step's local time, or its offset from the start of the horizon
    without usable timestamps."""
    if timestamps and len(timestamps) == len(dt):
        # Go stamps RFC3339Nano, fromisoformat takes at most six fractional digits
        return [datetime.fromisoformat(re.sub(r'\.\d+', '', ts)).strftime('%Y-%m-%d %H:%M') for ts in timestamps]

    labels, elapsed = [], 0
    for step in dt:
        labels.append(f'+{int(elapsed // 3600)}:{int(elapsed % 3600 // 60):02d}')
        elapsed += step
    return labels


def kwh(wh):
    return f'{wh / 1000:.1f}'


def per_kwh(per_wh):
    return f'{per_wh * 1000:.3f}'


# evcc battery detail types as the model should call them
KINDS = {'vehicle': 'vehicle', 'battery': 'home battery', 'loadpoint': 'charging point'}


def digest(payload):
    """Render the plan as text the model can reason over: one line per vehicle or battery with its bounds
    and one row per step, the same timeline the optimizer page shows."""
    req, res = payload['req'], payload['res']
    details = payload.get('details') or {}
    currency = payload.get('currency') or 'EUR'

    bat_details = details.get('batteryDetails') or []
    titles, batteries = [], []
    for i, bat in enumerate(req['batteries']):
        detail = at(bat_details, i, {})
        kind = KINDS.get(detail.get('type'), 'battery')
        titles.append(detail.get('title') or detail.get('name') or f'{kind} {i}')
        extras = [
            'may charge from grid' if bat.get('charge_from_grid') else 'no grid charging',
            'may discharge to grid' if bat.get('discharge_to_grid') else 'no grid discharge',
            f"remaining energy valued {per_kwh(bat['p_a'])} {currency}/kWh",
            'has charge demands' if any(v > 0 for v in bat.get('p_demand') or []) else '',
            'has soc goals' if any(v > 0 for v in bat.get('s_goal') or []) else '',
        ]
        batteries.append(
            f"{kind} \"{titles[i]}\" ({detail.get('capacity', kwh(bat.get('s_capacity', bat['s_max'])))} kWh): "
            f"soc {kwh(bat['s_initial'])} kWh within {kwh(bat['s_min'])}..{kwh(bat['s_max'])} kWh, "
            f"charge {kwh(bat['c_min'])}..{kwh(bat['c_max'])} kW, "
            + (f"discharge {kwh(bat['d_max'])} kW, " if bat['d_max'] > 0 else 'no discharge, ')
            + ', '.join(e for e in extras if e))

    ts = req['time_series']
    labels = step_labels(ts['dt'], details.get('timestamp'))
    horizon = round(sum(ts['dt']) / 3600)
    strategy = (req.get('strategy') or {}).get('charging_strategy') or 'none'
    objective = res.get('objective_value')

    lines = [
        f"Plan status {res['status']}, objective {objective if objective is not None else 'none'} {currency}, "
        f"{len(ts['dt'])} steps over {horizon} h, charging strategy {strategy}.",
        'Vehicles and batteries:',
        *batteries,
        f'Plan per step, step start time and length in minutes, energies in kWh per step, prices in {currency}/kWh, '
        'vehicle and battery columns are charge/discharge/soc at the end of the step:',
        ' | '.join(['step', 'minutes', 'import price', 'feed-in price', 'pv', 'load', 'grid import/export', *titles]),
    ]
    for i, label in enumerate(labels):
        lines.append(' | '.join([
            label, str(round(ts['dt'][i] / 60)), per_kwh(at(ts['p_N'], i)), per_kwh(at(ts['p_E'], i)),
            kwh(at(ts['ft'], i)), kwh(at(ts['gt'], i)),
            f"{kwh(at(res.get('grid_import'), i))}/{kwh(at(res.get('grid_export'), i))}",
            *(f"{kwh(at(b.get('charging_power'), i))}/{kwh(at(b.get('discharging_power'), i))}/{kwh(at(b.get('state_of_charge'), i))}"
              for b in res.get('batteries') or []),
        ]))

    return '\n'.join(lines)


def chat(settings, messages):
    """One chat completion against the Azure OpenAI v1 API. Returns the text and the usage."""
    body = {
        'model': settings.azure_openai_deployment,
        'messages': messages,
        # the plan is a table to read, not a puzzle: low effort answers in a few seconds
        'reasoning_effort': 'low',
        'max_completion_tokens': 1500,
    }
    request = urllib.request.Request(
        settings.azure_openai_endpoint.rstrip('/') + '/openai/v1/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'api-key': settings.azure_openai_key},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=CHAT_TIMEOUT) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors='replace')[:200]
        raise ExplainError(f'model returned {e.code}: {detail}') from e
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise ExplainError(f'model unreachable: {e}') from e

    try:
        return data['choices'][0]['message']['content'], data.get('usage') or {}
    except (KeyError, IndexError, TypeError) as e:
        raise ExplainError('model returned no answer') from e


def explain(payload, settings):
    """Explain a plan in the payload's language. Returns the markdown text and the token usage."""
    language = payload.get('language') or 'en'
    messages = [
        {'role': 'system', 'content': f'{SYSTEM_PROMPT}\nAnswer in the language with IETF tag {language}.'},
        {'role': 'user', 'content': f'Explain this plan.\n\n{digest(payload)}'},
    ]
    return chat(settings, messages)
