"""Plan explanations: the plan rendered as text and handed to a chat model."""
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta

# the prompt is not a request parameter: the service would be an open relay to the model otherwise
SYSTEM_PROMPT = """You explain an energy plan computed by evcc to the owner of the home. They are not technical: write for a \
neighbour, not an engineer. Explain what the plan does and why, in plain words.

# What you get
A plan for a home with its vehicles and home batteries. Read from the data what the home has: solar may be zero \
throughout, the grid price may be fixed or change over time, the export price may be zero. \
It starts with the current time, how long the plan runs, the plan status, the price units, the grid limits and a \
line per vehicle and home battery with everything known about it: capacity, charge level now, limits, power, \
whether it may charge from the grid, its charging goal and times it must charge. Then a table with one row per \
time slot: start time, length in minutes, import price, export price, solar forecast, household load forecast, \
grid import/export, and per vehicle and home battery the energy charged/discharged and the charge level in % at \
the end of the slot.
- Energies in the table are kWh per slot, not power: 1 kWh in a 15 minute slot means 4 kW. Slots are 15 minutes, \
the first one only covers the rest of the current quarter hour, slots after 48 hours are an hour long.
- Times are already relative to now: today, tomorrow, or the weekday.
- Charged energy is measured at the house, the charge level rises less: each charge and each discharge loses about \
10%, so storing energy and using it later loses about 19%.

# How the plan is made
evcc looks ahead over the forecasts for solar, household load and prices, and picks the plan with the lowest \
energy cost. It trusts the forecasts as they are, the plan changes when they change.
- Storing grid energy only pays when the price later is about 25% higher, because of the losses.
- Energy left in a vehicle or battery at the end of the plan is valued just below the cheapest import price. So \
storage soaks up surplus solar even without a goal, and buying grid energy just to keep it never pays.
- Exporting earns the export price, which is usually far below the import price: using or storing solar is \
better than exporting it, unless everything is full.
- A charging goal is a charge level a vehicle should reach by a time, set by the owner in a charging plan. \
"Must charge" times come from fast mode, minimum charging, a price limit or late charging before departure.
- Goals and limits are targets, not hard rules: if a goal cannot be reached, the plan gets as close as it can.
- "No grid charging" means it only charges from surplus solar. Vehicles never discharge. A minimum charge power \
means it charges at least that much or not at all.
- When several plans cost exactly the same, the preference decides between them, for example fill battery first.
- Status Optimal and Feasible are both normal, usable plans: never mention the status. Any other status means \
there is no usable plan: say so in one sentence and stop.

# Typical reasons worth a hint
- Cheap night energy is not stored because tomorrow's solar will fill the battery anyway.
- A battery keeps its charge for the expensive evening instead of covering cheap hours.
- Solar is exported because every vehicle and battery is full.
- A battery empties early to make room for negative prices or for solar that could not be exported.
- Several hours cost the same, so when exactly charging happens is arbitrary.
- A charging goal cannot be reached in time, or the grid limit is reached.
- An unusual spike in the household load, possibly a measurement glitch.
- Little solar or a big price gap between night and evening explains most of the plan.

# How to answer
Write two or three sentences on what the coming days look like and the idea of the plan, mentioning solar and \
price changes only where the data has them. Say what is decisive or surprising. When little happens, one sentence is enough. Then one \
bullet per vehicle and home battery, in the order given: `- **<name>**: <one sentence>`, saying what it does when, \
why, and where it ends. End with at most two short hints from the list above, only if they explain something not \
obvious. Separate the intro, the list and the hints with a blank line. No headings, no tables. Stay under 150 words.
- Use the times as given: "tomorrow 14:00", "Saturday 10:00", "tonight", "from 18:00". Never write dates or years. \
Ranges as "10:00 to 13:00".
- Energy in kWh with one decimal, charge levels in %, prices in the unit given. Use the decimal separator of the \
language you answer in.
- Never use dashes as punctuation, write a full stop or a comma instead.
- Never use: slot, step, horizon, objective, solver, SoC, Wh, loadpoint.
- Call every vehicle and home battery by its name. Only cite numbers that are in the plan, do not compute new ones."""

CHAT_TIMEOUT = 45  # seconds, below the gunicorn request timeout


class ExplainError(Exception):
    """The model could not be asked."""


def at(series, i, default=0):
    return series[i] if series and i < len(series) else default


def step_labels(dt, timestamps=None):
    """One label per step start plus one for the end of the plan: the local time relative to the first step,
    or the offset from the start without usable timestamps."""
    if timestamps and len(timestamps) == len(dt):
        # Go stamps RFC3339Nano, fromisoformat takes at most six fractional digits
        times = [datetime.fromisoformat(re.sub(r'\.\d+', '', ts)) for ts in timestamps]
        times.append(times[-1] + timedelta(seconds=dt[-1]))
        today = times[0].date()

        def day(t):
            return {0: 'today', 1: 'tomorrow'}.get((t.date() - today).days, t.strftime('%A'))

        return [f'{day(t)} {t:%H:%M}' for t in times]

    labels, elapsed = [], 0
    for step in [*dt, 0]:
        labels.append(f'+{int(elapsed // 3600)}:{int(elapsed % 3600 // 60):02d}')
        elapsed += step
    return labels


def kwh(wh):
    return f'{wh / 1000:.1f}'


# energy prices in the subunit evcc shows them in, the source of truth is ENERGY_PRICE_IN_SUBUNIT in
# https://github.com/evcc-io/evcc/blob/master/assets/js/mixins/formatter.ts
PRICE_SUBUNIT = {
    'EUR': 'ct', 'CHF': 'ct', 'GBP': 'p', 'USD': '¢', 'CAD': '¢', 'BRL': '¢', 'AUD': 'c', 'NZD': 'c', 'ZAR': 'c',
    'NOK': 'øre', 'DKK': 'øre', 'SEK': 'öre', 'PLN': 'gr', 'BGN': 'st', 'ILS': 'ag', 'TRY': 'krş',
}


def price_format(currency):
    """The unit prices are given in and a formatter from price per Wh to it."""
    if currency in PRICE_SUBUNIT:
        return PRICE_SUBUNIT[currency], lambda per_wh: f'{per_wh * 1e5:.1f}'
    return currency, lambda per_wh: f'{per_wh * 1000:.3f}'


def ranges(series, labels):
    """Start to end label of every run of steps with a positive value."""
    runs, start = [], None
    for i, v in enumerate([*(series or []), 0]):
        if v > 0 and start is None:
            start = i
        elif v <= 0 and start is not None:
            runs.append(f'{labels[start]} to {labels[i]}')
            start = None
    return runs


STRATEGIES = {
    'charge_before_export': 'fill battery first',
    'attenuate_demand_peaks': 'reduce grid import peaks',
    'attenuate_feedin_peaks': 'reduce grid export peaks',
    'attenuate_grid_peaks': 'reduce grid peaks',
}

DEMANDS = {'heating': 'heating', 'unmodelled': 'charging point without vehicle data'}


def capacity_wh(bat, detail):
    return bat.get('s_capacity') or (detail['capacity'] * 1000 if detail.get('capacity') else bat['s_max']) or 1


def battery_line(bat, detail, i, labels, capacity):
    kind = 'vehicle' if detail.get('type') == 'vehicle' else 'home battery'
    title = detail.get('title') or detail.get('name') or f'{kind} {i + 1}'

    def pct(wh):
        return f'{round(wh / capacity * 100)}%'

    facts = [
        f"capacity {kwh(capacity)} kWh",
        f"now {pct(bat['s_initial'])} ({kwh(bat['s_initial'])} kWh)",
        f"kept between {pct(bat['s_min'])} and {pct(bat['s_max'])}",
        f"charges up to {kwh(bat['c_max'])} kW" + (f", at least {kwh(bat['c_min'])} kW or not at all" if bat['c_min'] > 0 else ''),
        f"discharges up to {kwh(bat['d_max'])} kW" if bat['d_max'] > 0 else 'never discharges',
        'may charge from the grid' if bat.get('charge_from_grid') else 'no grid charging, only surplus solar',
    ]
    if bat['d_max'] > 0:
        facts.append('may discharge into the grid' if bat.get('discharge_to_grid') else 'does not discharge into the grid')
    if bat.get('c_active'):
        facts.append('charging right now')
    facts += [f'charging goal {pct(goal)} by {labels[t + 1]}' for t, goal in enumerate(bat.get('s_goal') or []) if goal > 0]
    facts += [f'must charge {run}' for run in ranges(bat.get('p_demand'), labels)]
    return title, f'{kind} "{title}": ' + ', '.join(facts)


def digest(payload):
    """Render the plan as text the model can reason over: everything known per vehicle and battery, then one row
    per step, the same timeline the optimizer page shows."""
    req, res = payload['req'], payload['res']
    details = payload.get('details') or {}
    unit, price = price_format(payload.get('currency') or 'EUR')
    ts = req['time_series']
    labels = step_labels(ts['dt'], details.get('timestamp'))

    titles, devices, capacities = [], [], []
    for i, bat in enumerate(req['batteries']):
        detail = at(details.get('batteryDetails'), i, {})
        capacities.append(capacity_wh(bat, detail))
        title, line = battery_line(bat, detail, i, labels, capacities[-1])
        titles.append(title)
        devices.append(line)

    grid = req.get('grid') or {}
    violations = res.get('limit_violations') or {}
    notes = [
        f"grid import limited to {kwh(grid['p_max_imp'])} kW" if grid.get('p_max_imp') else '',
        f"grid export limited to {kwh(grid['p_max_exp'])} kW" if grid.get('p_max_exp') else '',
        'the plan needs more grid import than the limit allows' if violations.get('grid_import_limit_exceeded') else '',
        'the plan reaches the grid export limit, surplus solar beyond it is lost' if violations.get('grid_export_limit_hit') else '',
    ]
    demands = [f"{DEMANDS[d['type']]} \"{d.get('title') or d['type']}\"" for d in details.get('demandDetails') or [] if d.get('type') in DEMANDS]
    strategy = STRATEGIES.get((req.get('strategy') or {}).get('charging_strategy'), 'no preference')

    lines = [
        f"Now: {labels[0]}. The plan runs until {labels[-1]}. Status {res['status']}. Preference: {strategy}.",
        f"Prices in {unit}/kWh, energies in kWh. Charging and discharging efficiency {req.get('eta_c', 0.95)} and {req.get('eta_d', 0.95)}.",
        *(f'{n}.' for n in notes if n),
        *([f"The household load includes {', '.join(demands)}."] if demands else []),
        'Vehicles and home batteries:',
        *devices,
        'Plan per time slot, the vehicle and battery columns are charged/discharged/charge level in % at the end of the slot:',
        ' | '.join(['time', 'minutes', 'import price', 'export price', 'solar', 'load', 'grid import/export', *titles]),
    ]
    for i, label in enumerate(labels[:-1]):
        lines.append(' | '.join([
            label, str(round(ts['dt'][i] / 60)), price(at(ts['p_N'], i)), price(at(ts['p_E'], i)),
            kwh(at(ts['ft'], i)), kwh(at(ts['gt'], i)),
            f"{kwh(at(res.get('grid_import'), i))}/{kwh(at(res.get('grid_export'), i))}",
            *(f"{kwh(at(b.get('charging_power'), i))}/{kwh(at(b.get('discharging_power'), i))}/"
              f"{round(at(b.get('state_of_charge'), i) / capacity * 100)}"
              for b, capacity in zip(res.get('batteries') or [], capacities)),
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
    text, usage = chat(settings, messages)
    # the prompt forbids dashes, models still slip one in now and then
    return text.replace(' — ', ', ').replace('—', ', ').replace('–', '-'), usage
