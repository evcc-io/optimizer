"""Plan explanations: the plan folded into clock hours and handed to a chat model."""
import json
import re
import urllib.error
import urllib.request
from datetime import datetime

SYSTEM_PROMPT = """You explain the schedule of the evcc home energy optimizer to the owner of the home.
The optimizer solves a mixed integer linear program over the horizon. It minimises grid import cost minus
feed-in revenue minus the value of the energy left in each battery at the end of the horizon (p_a per kWh),
subject to the power balance, the battery bounds and efficiencies, grid limits, charge demands and soc goals.
A charging strategy only chooses between schedules of equal cost. Status Optimal is the proven cost optimum,
Feasible is a schedule the solver stopped on at its time limit, Infeasible means the constraints contradict
each other and no schedule exists.
Explain why the plan looks the way it does: when and why each battery charges or discharges, how prices, solar
and load drive that, and what state it ends in. Point out anything surprising. Be concrete, cite hours and
numbers from the plan. Use markdown with short paragraphs, no tables. Stay under 250 words."""

CHAT_TIMEOUT = 45  # seconds, below the gunicorn request timeout


class ExplainError(Exception):
    """The model could not be asked."""


def at(series, i, default=0):
    return series[i] if series and i < len(series) else default


def hour_label(timestamp):
    # Go stamps RFC3339Nano, fromisoformat takes at most six fractional digits
    start = datetime.fromisoformat(re.sub(r'\.\d+', '', timestamp))
    return start.strftime('%Y-%m-%d %H:00')


def hours(req, res, timestamps=None):
    """Fold the steps of a plan into clock hours: energies summed, prices averaged by duration, soc
    taken at the end of the hour. Hourly steps map one to one. Without usable timestamps the hours
    count from the start of the horizon."""
    ts = req['time_series']
    dt = ts['dt']
    batteries = res.get('batteries') or []
    if not timestamps or len(timestamps) != len(dt):
        timestamps = None

    rows = []
    elapsed = 0
    for i, step in enumerate(dt):
        label = hour_label(timestamps[i]) if timestamps else f'+{int(elapsed // 3600)}h'
        elapsed += step

        if not rows or rows[-1]['hour'] != label:
            rows.append({'hour': label, 'seconds': 0, 'price': 0, 'feed_in': 0, 'pv': 0, 'load': 0,
                         'grid_import': 0, 'grid_export': 0,
                         'batteries': [{'charge': 0, 'discharge': 0, 'soc': 0} for _ in batteries]})
        row = rows[-1]

        row['seconds'] += step
        row['price'] += at(ts['p_N'], i) * step
        row['feed_in'] += at(ts['p_E'], i) * step
        row['pv'] += at(ts['ft'], i)
        row['load'] += at(ts['gt'], i)
        row['grid_import'] += at(res.get('grid_import'), i)
        row['grid_export'] += at(res.get('grid_export'), i)
        for b, bat in enumerate(batteries):
            entry = row['batteries'][b]
            entry['charge'] += at(bat.get('charging_power'), i)
            entry['discharge'] += at(bat.get('discharging_power'), i)
            entry['soc'] = at(bat.get('state_of_charge'), i, entry['soc'])

    for row in rows:
        if row['seconds']:
            row['price'] /= row['seconds']
            row['feed_in'] /= row['seconds']

    return rows


def kwh(wh):
    return f'{wh / 1000:.1f}'


def per_kwh(per_wh):
    return f'{per_wh * 1000:.3f}'


def digest(payload):
    """Render the plan as text the model can reason over: one line per battery with its bounds and one
    row per hour. A few KB for a two day horizon, where the raw arrays run to hundreds."""
    req, res = payload['req'], payload['res']
    details = payload.get('details') or {}
    currency = payload.get('currency') or 'EUR'

    bat_details = details.get('batteryDetails') or []
    titles = [(at(bat_details, i, {}).get('title') or at(bat_details, i, {}).get('name') or f'battery {i}')
              for i in range(len(req['batteries']))]

    batteries = []
    for i, bat in enumerate(req['batteries']):
        detail = at(bat_details, i, {})
        extras = [
            'may charge from grid' if bat.get('charge_from_grid') else 'no grid charging',
            'may discharge to grid' if bat.get('discharge_to_grid') else 'no grid discharge',
            f"remaining energy valued {per_kwh(bat['p_a'])} {currency}/kWh",
            'has charge demands' if any(v > 0 for v in bat.get('p_demand') or []) else '',
            'has soc goals' if any(v > 0 for v in bat.get('s_goal') or []) else '',
        ]
        batteries.append(
            f"{i}: {titles[i]} ({detail.get('type', 'battery')}, {detail.get('capacity', kwh(bat.get('s_capacity', bat['s_max'])))} kWh): "
            f"soc {kwh(bat['s_initial'])} kWh within {kwh(bat['s_min'])}..{kwh(bat['s_max'])} kWh, "
            f"charge {kwh(bat['c_min'])}..{kwh(bat['c_max'])} kW, discharge {kwh(bat['d_max'])} kW, "
            + ', '.join(e for e in extras if e))

    rows = hours(req, res, details.get('timestamp'))
    horizon = round(sum(r['seconds'] for r in rows) / 3600)
    strategy = (req.get('strategy') or {}).get('charging_strategy') or 'none'
    objective = res.get('objective_value')

    lines = [
        f"Plan status {res['status']}, objective {objective if objective is not None else 'none'} {currency}, "
        f"{len(req['time_series']['dt'])} steps over {horizon} h, charging strategy {strategy}.",
        'Batteries:',
        *batteries,
        f'Plan per hour, energies in kWh, prices in {currency}/kWh, battery columns are charge/discharge/soc at the end of the hour:',
        ' | '.join(['hour', 'import price', 'feed-in price', 'pv', 'load', 'grid import/export', *titles]),
    ]
    for r in rows:
        lines.append(' | '.join([
            r['hour'], per_kwh(r['price']), per_kwh(r['feed_in']), kwh(r['pv']), kwh(r['load']),
            f"{kwh(r['grid_import'])}/{kwh(r['grid_export'])}",
            *(f"{kwh(b['charge'])}/{kwh(b['discharge'])}/{kwh(b['soc'])}" for b in r['batteries']),
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
