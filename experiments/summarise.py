"""Summarise E15 results into results/E15_summary.md (latency with miss rate, paired differences). Run from the repository root."""
import json
import sys

NAMES = {
    'local': 'All-local',
    'plan': 'Static plan, sampled per class',
    'plan_refresh': 'Static plan re-planned every 20 slots (Stage 2)',
    'tan': 'Tan-style: re-solved every slot, z = 0, blind',
    'tan_z2': 'Tan-style with z = 2 margin, blind',
    'plan_rank': 'GIPA-B: static plan + per-task ranking',
    'gipa_plan': 'GIPA-A: static plan + operator',
    'gipa_refresh': 'GIPA-A on the Stage-2 refreshed plan',
    'tan_rank': 'Tan-style plan + per-task ranking',
    'tan_gipa': 'Tan-style plan + GIPA operator',
}


def table(res, scen, arms, Ns=(80, 160, 240, 320)):
    rows = ['| Policy | ' + ' | '.join(f'N = {n}' for n in Ns) + ' |', '| --- |' + ' --- |' * len(Ns)]
    for a in arms:
        cells = []
        for n in Ns:
            r = res['main'][f'{scen}|{n}'][a]
            cells.append(f"{r['lat']:.2f} ({100 * r['miss']:.1f}%)")
        rows.append(f'| {NAMES[a]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def paired(res, scen, keys, Ns=(80, 160, 240, 320)):
    rows = ['| Difference | metric | ' + ' | '.join(f'N = {n}' for n in Ns) + ' |', '| --- | --- |' + ' --- |' * len(Ns)]
    for k in keys:
        for metric, scale, unit in (('lat', 1, 'ms'), ('miss', 100, 'pp')):
            cells = []
            for n in Ns:
                d = res['paired'][f'{scen}|{n}'][f'{k}|{metric}']
                lo, hi = d['ci']
                sig = '' if lo <= 0 <= hi else '*'
                cells.append(f"{scale * d['mean']:+.2f} [{scale * lo:+.2f}, {scale * hi:+.2f}]{sig}")
            rows.append(f'| {k} | {unit} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def main():
    out = []
    files = dict(batch='results/res_tan_batch.json', carry='results/res_tan_carry.json', fluct='results/res_tan_fluct.json')
    res = {}
    for k, f in files.items():
        try:
            res[k] = json.load(open(f))
        except FileNotFoundError:
            pass
    arms = ['local', 'plan', 'tan', 'tan_z2', 'plan_rank', 'gipa_plan', 'tan_rank', 'tan_gipa']
    keys = ['gipa_plan-tan', 'plan_rank-tan', 'tan_gipa-tan', 'tan_rank-tan', 'plan-tan', 'tan_z2-tan',
            'tan_gipa-gipa_plan', 'tan_rank-plan_rank']
    for part, title in (('batch', 'Tasks independent across slots (batch)'), ('carry', 'Carried-over local work (own mode)')):
        if part not in res:
            continue
        out.append(f'## {title}\n')
        out.append(f"Tuned operator parameters: see `chosen` in the JSON. Run time {res[part]['seconds'] / 60:.0f} min.\n")
        for sc in ('homog', 'cov2', 'het25', 'het50', 'het50cov2', 'het50cov2p20'):
            out.append(f'### {sc}: mean latency in ms (overall deadline misses)\n')
            out.append(table(res[part], sc, arms))
            out.append('\nPaired differences, mean [95% bootstrap CI over 30 seeds]; * = CI excludes 0\n')
            out.append(paired(res[part], sc, keys))
            out.append('')
        fails = [r.get('solve_fail', 0) for k, v in res[part]['per_seed'].items() if '|tan' in k for r in v]
        out.append(f'Tan-style solver fallbacks: {100 * sum(fails) / len(fails):.3f}% of slots.\n')
    if 'fluct' in res:
        arms_f = ['local', 'plan', 'plan_refresh', 'tan', 'tan_z2', 'plan_rank', 'gipa_plan', 'gipa_refresh', 'tan_rank', 'tan_gipa']
        out.append('## Fluctuating fleet (0.6-1.4 x N every 10 slots, batch)\n')
        for sc in ('fluct_homog', 'fluct_het50'):
            out.append(f'### {sc}\n')
            out.append(table(res['fluct'], sc, arms_f))
            out.append('\nPaired differences against the Tan-style plan\n')
            out.append(paired(res['fluct'], sc, keys))
            out.append('')
    open('results/E15_summary.md', 'w').write('\n'.join(out))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
