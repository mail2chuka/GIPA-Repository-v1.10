"""
compile_carry.py
================
Experiment E13 summary: the slotted experiments with carried-over local work
against the archived idle-processor results.

Reads the idle-processor result files in results/ and their reruns in
results/carry_own/ and results/carry_blind/ (written by ``run_carry.sh``),
and writes, for each manuscript section, one table per regime with the mean
latency (ms) and the safety / background deadline-miss rates of the main
policies under the three local models:

    idle   the paper's model: a task kept local takes L_k from arrival
    own    unfinished local work carries over to the next slot, and each
           vehicle uses its own backlog in its decisions (the specified GIPA)
    blind  unfinished local work carries over; decisions ignore it

A mode whose files are missing is shown as '-'. A plan that is infeasible
under a model is shown as 'infeasible'. res_main.json and res_sc.json for
the idle model come from results/ (exp.py writes them to the working
directory; the archived copies are in results/).

Output
------
results/carry_summary.json  {section: {regime: {N: {policy: {mode: means}}}}}
results/carry_summary.md    the same as Markdown tables

Run (from the repository root)
------------------------------
    python experiments/compile_carry.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MODES = ['idle', 'own', 'blind']
DIRS = {'idle': RES, 'own': RES / 'carry_own', 'blind': RES / 'carry_blind'}
FLEETS = ['80', '160', '240', '320']


def load(mode, name):
    p = DIRS[mode] / name
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def pick(m):
    if not isinstance(m, dict) or m.get('lat') is None:
        return None

    def val(k):
        v = m.get(k)
        return v['mean'] if isinstance(v, dict) else v   # Q-learning grid stores {mean, ci95, ...}
    return {k: val(k) for k in ['lat', 'mS', 'mB', 'offS', 'offB']}


def grid(getter, regimes, fleets, policies):
    """Apply getter(data, regime, N, policy) to each mode's data."""
    out = {}
    for reg in regimes:
        out[reg] = {}
        for N in fleets:
            out[reg][N] = {}
            for pol in policies:
                out[reg][N][pol] = {}
                for mode, data in getter['data'].items():
                    if data is None:
                        out[reg][N][pol][mode] = '-'
                        continue
                    try:
                        v = getter['fn'](data, reg, N, pol)
                    except (KeyError, IndexError, TypeError):
                        v = None
                    v = pick(v)
                    out[reg][N][pol][mode] = v if v is not None else 'infeasible'
    return out


def section(fname, fn, regimes, fleets, policies):
    data = {m: load(m, fname) for m in MODES}
    return grid(dict(data=data, fn=fn), regimes, fleets, policies)


def build():
    S = {}
    S['VII-A main comparison (affine), res_main'] = section(
        'res_main.json', lambda d, r, N, p: d[r][N][p], ['A', 'B'], FLEETS,
        ['local', 'gipa', 'msoft_prior', 'msoft', 'rbr', 'price', 'entreg', 'indep', 'greedy', 'ql'])
    S['VII-D ablation (affine), res_main'] = section(
        'res_main.json', lambda d, r, N, p: d[r][N][p], ['A'], FLEETS,
        ['gipa', 'gipa_noamp', 'gipa_nodiff', 'gipa_fixed', 'gipa_binary', 'gipa_noprior'])
    S['VII-A planned prior (affine), res_planner'] = section(
        'res_planner.json', lambda d, r, N, p: d['main'][r][N][p], ['A', 'B'], FLEETS,
        ['local', 'gipa', 'plan_z0', 'plan_z2'])
    S['VII-A Eq. (5) model, res_calibration'] = section(
        'res_calibration.json', lambda d, r, N, p: d['main'][r][N][p], ['A', 'B'], FLEETS,
        ['local', 'gipa|class', 'msoft_prior|class', 'rbr', 'plan|class', 'plan_z2|class'])
    S['VII-C RSU capacity, admission control, res_capacity'] = section(
        'res_capacity.json', lambda d, r, N, p: d['main'][r][N][p]['ac'], ['A', 'B'], FLEETS,
        ['local', 'gipa', 'msoft_prior', 'rbr', 'plan'])

    def pred(d, r, N, p):
        name, pol = p.split(':')
        for c in d['conditions']:
            if c['name'] == name and str(c['N']) == N:
                return c['results'][pol]
        raise KeyError(p)
    S['VII-B prediction error (Regime A, N = 160), res_prediction'] = section(
        'res_prediction.json', pred, ['A'], ['160'],
        [f'{c}:{p}' for c in ['baseline', 'both_sigma0.30', 'both_sigma0.30_guard0.10', 'drift0.2_ewma']
         for p in ['local', 'gipa', 'msoft_prior', 'plan']])
    S['VII-B observability (Regime A), res_prediction_obs'] = section(
        'res_prediction_obs.json', pred, ['A'], ['80', '160', '240'],
        [f'{c}:{p}' for c in ['oracle', 'noise0.3_guard0.0', 'drift0.2_ewma_total_debiased']
         for p in ['plan', 'gipa_plan', 'gipa_plan_z2', 'gipa']])

    def asym(d, r, N, p):
        scen, model = r.split('/')
        return d['main'][f'{scen}|{model}'][N][p]
    S['VII-D asymmetric RSUs, res_asym'] = section(
        'res_asym.json', asym, ['sym/A|affine', 'fast1/A|affine', 'fast1/A|ps', 'fast1/B|affine'], FLEETS,
        ['local', 'plan', 'la', 'gipa|capacity', 'msoft_prior|capacity', 'rbr'])

    def la(d, r, N, p):
        return d['main'][r][N][p]
    S['VII-G learning automata, res_la'] = section(
        'res_la.json', la, ['A|affine|500', 'A|ps|500', 'B|affine|500'], FLEETS,
        ['local', 'gipa', 'plan', 'best', 'best_perdensity'])

    def ql(d, r, N, p):
        return d['grid'][f'{r}|{p}'][N]
    S['VII-G Q-learning, res_qlearning'] = section(
        'res_qlearning.json', ql, ['A|affine', 'A|ps', 'B|affine'], FLEETS,
        ['gamma0_N160', 'gamma0_perdensity', 'gamma09_N160'])

    def het(d, r, N, p):
        return d['main'][r][N][p]
    S['VII-I heterogeneous tasks, res_hetero'] = section(
        'res_hetero.json', het, ['homog', 'het50', 'het50cov2p20'], FLEETS,
        ['local', 'gipa', 'plan', 'plan_mask', 'gipa_plan', 'la'])
    return S


def cell(v):
    if v == '-' or v == 'infeasible':
        return v
    s = f"{v['lat']:.2f}"
    if v['mS'] is not None and v['mB'] is not None:
        s += f" ({100 * v['mS']:.1f}/{100 * v['mB']:.1f})"
    return s


def markdown(S):
    lines = ['# E13: slotted experiments with carried-over local work', '',
             'Mean latency in ms (safety / background deadline misses, %). '
             'idle: archived model; own: carry-over, decisions use the own backlog; '
             'blind: carry-over, decisions ignore it.', '']
    for sec, tab in S.items():
        lines += [f'## {sec}', '']
        for reg, byN in tab.items():
            lines += [f'### {reg}', '', '| N | policy | ' + ' | '.join(MODES) + ' |',
                      '|---|---|' + '---|' * len(MODES)]
            for N, byP in byN.items():
                for pol, byM in byP.items():
                    lines.append(f'| {N} | {pol} | ' + ' | '.join(cell(byM[m]) for m in MODES) + ' |')
            lines.append('')
    return '\n'.join(lines)


def main():
    S = build()
    with open(RES / 'carry_summary.json', 'w') as fh:
        json.dump(S, fh, indent=1)
    with open(RES / 'carry_summary.md', 'w') as fh:
        fh.write(markdown(S))
    print('wrote', RES / 'carry_summary.json', 'and', RES / 'carry_summary.md')


if __name__ == '__main__':
    main()
