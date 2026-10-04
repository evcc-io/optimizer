"""Benchmark the departure probabilities against a reference revision.

Runs every stored test case three ways: through the reference revision, through the working tree as
stored, and with a 30 percent departure chance spread over hours 6 to 12 on every battery that cannot
discharge. Reports solve time, model size and whether the schedule moved.

    uv run python tools/bench_departure.py [--rev main] [--repeat 3]
"""
import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))

from bench_peak_leveling import build, load_reference  # noqa: E402

import optimizer.optimizer as new  # noqa: E402  the working tree under test, needs the path above


def with_departure(req):
    req = copy.deepcopy(req)
    hours = np.cumsum(req['time_series']['dt']) / 3600
    window = (hours > 6) & (hours <= 12)
    if not window.any():
        return None
    p = np.where(window, 0.3 / window.sum(), 0.0)
    for bat in req['batteries']:
        if bat['d_max'] == 0:
            bat['p_departure'] = p.tolist()
    return req


def without_departure(req):
    """the request as the reference revision understands it"""
    req = copy.deepcopy(req)
    for bat in req['batteries']:
        bat.pop('p_departure', None)
    return req


def build_new(req):
    opt = build(new, without_departure(req))
    for bat, cfg in zip(req['batteries'], opt.batteries):
        cfg.p_departure = bat.get('p_departure')
    return new.Optimizer(opt.strategy, opt.grid, opt.batteries, opt.time_series, opt.eta_c, opt.eta_d, opt.M)


def run(make, req, repeat):
    best, res, opt = None, None, None
    for _ in range(repeat):
        o = make(req)
        started = time.perf_counter()
        r = o.solve()
        best = min(best or 1e9, time.perf_counter() - started)
        res, opt = r, o
    return {'t': best, 'res': res, 'rows': len(opt.problem.constraints), 'cols': len(opt.problem.variables())}


def charging(res):
    return np.array([b['charging_power'] for b in res['batteries']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rev', default='main', help='revision to compare against')
    ap.add_argument('--repeat', type=int, default=3, help='solves per case, fastest counts')
    args = ap.parse_args()
    ref = load_reference(args.rev)

    print(f'{"case":46s} {"t_ref":>6s} {"t_new":>6s} {"t_dep":>6s} {"rows":>12s} {"cols":>10s}  departure')
    totals = np.zeros(3)
    for path in sorted((ROOT / 'test_cases').glob('*.json')):
        req = json.loads(path.read_text())['request']
        a = run(lambda r: build(ref, without_departure(r)), req, args.repeat)
        b = run(build_new, req, args.repeat)
        dep_req = with_departure(req)
        dep = run(build_new, dep_req, args.repeat) if dep_req else None
        totals += [a['t'], b['t'], dep['t'] if dep else 0]
        # a stored case with departure probabilities is a different model for the reference by design
        same = (np.allclose(charging(a['res']), charging(b['res']), atol=1)
                or np.isclose(a['res']['objective_value'], b['res']['objective_value'], atol=1e-4)
                or without_departure(req) != req)
        dep_moved = dep and not np.allclose(charging(b['res']), charging(dep['res']), atol=1)
        print(f'{path.stem:46s} {a["t"]:6.3f} {b["t"]:6.3f} {dep["t"] if dep else 0:6.3f} '
              f'{a["rows"]:5d}->{(dep or b)["rows"]:5d} {a["cols"]:4d}->{(dep or b)["cols"]:4d}  '
              f'{"moved" if dep_moved else "same" if dep else "-"}{"" if same else "  REF DIFFERS"}')
    print(f'{"total":46s} {totals[0]:6.3f} {totals[1]:6.3f} {totals[2]:6.3f}')


if __name__ == '__main__':
    main()
