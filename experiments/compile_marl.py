"""Summarise E14 (independent DQN and VDN) for the manuscript and the Supplementary Material.

Reads results/res_marl_value.json (prior-inclusive mask, N = 80-320) and results/res_marl_value_dmask.json
(deadline-only mask, N = 240 and 320); writes results/marl_summary.json with the numbers quoted in the paper
(Table S20.1 rows, Section VII-C) and the sentence used in Supplementary Section S10.5.
Run from the repository root: python -I experiments/compile_marl.py
"""
import json

d = json.load(open('results/res_marl_value.json'))
dm = json.load(open('results/res_marl_value_dmask.json'))
out = {}
for meth in ('idqn', 'vdn'):
    g = d['methods'][meth]['160']
    m, e = g['mean'], g['mean_eps_greedy']
    out[meth] = dict(
        lat160=round(m['lat'], 2), range160=[round(g['lat_min'], 2), round(g['lat_max'], 2)], fair160=round(m['fair'], 2),
        miss160=[round(100 * m['mS'], 2), round(100 * m['mB'], 2)], off160=round(m['off'], 3),
        eps_greedy_lat160=round(e['lat'], 2), rules160=[r['rule'] for r in g['runs']],
        lat80=round(d['methods'][meth]['80']['mean']['lat'], 2),
        lat_dmask={N: round(dm['methods'][meth][N]['mean']['lat'], 2) for N in ('240', '320')},
        safety_miss_dmask={N: round(100 * dm['methods'][meth][N]['mean']['mS'], 2) for N in ('240', '320')})
i, v = out['idqn'], out['vdn']
out['sentence'] = (
    f'At N = 160 the tuned execution rule gives {i["lat160"]:.2f} ms for independent DQN (training runs '
    f'{i["range160"][0]:.2f}–{i["range160"][1]:.2f} ms) and {v["lat160"]:.2f} ms for VDN ({v["range160"][0]:.2f}–'
    f'{v["range160"][1]:.2f} ms); with ε-greedy execution they give {i["eps_greedy_lat160"]:.2f} and '
    f'{v["eps_greedy_lat160"]:.2f} ms. At N = 80 they give {i["lat80"]:.2f} and {v["lat80"]:.2f} ms, and with the '
    f'deadline-only mask at N = 240 and 320 independent DQN gives {i["lat_dmask"]["240"]:.2f} and {i["lat_dmask"]["320"]:.2f} ms '
    f'and VDN {v["lat_dmask"]["240"]:.2f} and {v["lat_dmask"]["320"]:.2f} ms.')
json.dump(out, open('results/marl_summary.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
