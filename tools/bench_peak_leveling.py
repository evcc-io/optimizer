"""Benchmark the grid peak leveling terms against the previous formulation.

Runs every stored test case through the working tree and through a reference revision of
optimizer.py, and reports solve time, model size and how level the resulting grid profile is.

    uv run python tools/bench_peak_leveling.py [--rev main] [--repeat 3]
"""
import argparse
import importlib
import json
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import optimizer.optimizer as new  # noqa: E402  the working tree under test, needs the path above


def load_reference(rev):
    """import optimizer.py from a git revision as a standalone module"""
    src = subprocess.run(['git', 'show', f'{rev}:src/optimizer/optimizer.py'],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    tmp = pathlib.Path(tempfile.mkdtemp())
    pkg = tmp / 'reference'
    pkg.mkdir()
    (pkg / '__init__.py').write_text('')
    (pkg / 'settings.py').write_text((ROOT / 'src' / 'optimizer' / 'settings.py').read_text())
    (pkg / 'optimizer.py').write_text(src)
    sys.path.insert(0, str(tmp))
    return importlib.import_module('reference.optimizer')


def build(mod, req):
    s = req.get('strategy', {})
    g = req.get('grid', {})
    return mod.Optimizer(
        strategy=mod.OptimizationStrategy(
            charging_strategy=s.get('charging_strategy', 'none'),
            discharging_strategy=s.get('discharging_strategy', 'none')),
        grid=mod.GridConfig(p_max_imp=g.get('p_max_imp'), p_max_exp=g.get('p_max_exp'),
                            prc_p_exc_imp=g.get('prc_p_exc_imp')),
        batteries=[mod.BatteryConfig(
            charge_from_grid=b.get('charge_from_grid', False),
            discharge_to_grid=b.get('discharge_to_grid', False),
            s_capacity=b.get('s_capacity', b['s_max']), s_min=b['s_min'], s_max=b['s_max'],
            s_initial=b['s_initial'], p_demand=b.get('p_demand'), s_goal=b.get('s_goal'),
            c_min=b['c_min'], c_max=b['c_max'], d_max=b['d_max'], p_a=b['p_a'],
            c_priority=b.get('c_priority', 0)) for b in req['batteries']],
        time_series=mod.TimeSeriesData(**req['time_series']),
        eta_c=req.get('eta_c', 0.95), eta_d=req.get('eta_d', 0.95), M=1e6)


def profiles(opt, res):
    dt = np.array(opt.time_series.dt, float)
    imp = np.array(res['grid_import'], float)
    if res['grid_import_overshoot'] and not opt.is_grid_demand_rate_active:
        imp = imp + np.array(res['grid_import_overshoot'], float)
    exp = np.array(res['grid_export'], float)
    if res['grid_export_overshoot']:
        exp = exp + np.array(res['grid_export_overshoot'], float)
    return {'imp': imp * 3600 / dt, 'exp': exp * 3600 / dt}


def sd(p, dt):
    """time weighted standard deviation of the profile [W], the mean square deviation measure"""
    dt = np.asarray(dt, float)
    m = (p * dt).sum() / dt.sum()
    return float(np.sqrt(((p - m) ** 2 * dt).sum() / dt.sum()))


def run(mod, req, repeat):
    best, res, opt = None, None, None
    for _ in range(repeat):
        o = build(mod, req)
        started = time.perf_counter()
        r = o.solve()
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)
        res, opt = r, o
    return {'t': best, 'res': res, 'rows': len(opt.problem.constraints),
            'cols': len(opt.problem.variables()), 'opt': opt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rev', default='main', help='revision to compare against')
    ap.add_argument('--repeat', type=int, default=3, help='solves per case, fastest counts')
    ap.add_argument('--strategy', default=None,
                    help='force this charging strategy on every case instead of the stored one')
    args = ap.parse_args()

    ref = load_reference(args.rev)

    print(f'{"case":46s} {"t_ref":>7s} {"t_new":>7s} {"rows":>11s} {"cols":>9s}  '
          f'{"sd_ref":>8s} {"sd_new":>8s} {"peak_ref":>8s} {"peak_new":>8s}  schedule')
    total_ref = total_new = 0.0
    for path in sorted((ROOT / 'test_cases').glob('*.json')):
        req = json.loads(path.read_text())['request']
        if args.strategy:
            req.setdefault('strategy', {})['charging_strategy'] = args.strategy
        a, b = run(ref, req, args.repeat), run(new, req, args.repeat)
        total_ref += a['t']
        total_new += b['t']
        sides = new.PEAK_STRATEGY_SIDES.get(req.get('strategy', {}).get('charging_strategy'), ())
        pa, pb = profiles(a['opt'], a['res']), profiles(b['opt'], b['res'])
        dt = req['time_series']['dt']
        same = all(np.allclose(pa[s], pb[s], atol=1) for s in ('imp', 'exp'))
        sd_ref = ' '.join(f'{sd(pa[s], dt):.0f}' for s in sides) or '-'
        sd_new = ' '.join(f'{sd(pb[s], dt):.0f}' for s in sides) or '-'
        pk_ref = ' '.join(f'{pa[s].max():.0f}' for s in sides) or '-'
        pk_new = ' '.join(f'{pb[s].max():.0f}' for s in sides) or '-'
        print(f'{path.stem:46s} {a["t"]:7.3f} {b["t"]:7.3f} '
              f'{a["rows"]:5d}->{b["rows"]:5d} {a["cols"]:4d}->{b["cols"]:4d}  '
              f'{sd_ref:>8s} {sd_new:>8s} {pk_ref:>8s} {pk_new:>8s}  {"same" if same else "CHANGED"}')
    print(f'{"total":46s} {total_ref:7.3f} {total_new:7.3f}')


if __name__ == '__main__':
    main()
