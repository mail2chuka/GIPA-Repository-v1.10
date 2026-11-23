"""
diag_sla.py
===========
Experiment E16 diagnostics: how SLA-MA-PPO spreads load, at N = 160.

For every trained SLA-MA-PPO actor (results/sla/<setting>_<carry>_N160_r*.json)
the 30 evaluation episodes are replayed and, after the warm-up, each slot's
number of offloaded tasks, largest RSU load and Jain index of the RSU loads
are recorded. Reported per run: mean offloaded tasks per slot, its standard
deviation against the binomial value sqrt(N p (1 - p)) that independent
draws with the same mean share would give, the lag-1 autocorrelation of
the offloaded count (positive values mean the fleet stays high or low for
several slots, which independent sampling of a fixed plan cannot do), the
mean largest RSU load and the mean Jain index. The planned prior offloads
N x M x x_B background tasks per slot on average (results/res_planner.json).

Writes results/sla_diagnostics.json. Uses the same seeds and action
generator as the evaluation, so it reproduces the evaluation exactly.
Carried-over runs need GIPA_LOCAL_CARRY=own (run_e16.sh diag does both).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import sla_mappo as S  # noqa: E402
from gipa_core import CFG, LOCAL_CARRY, RNG_ACT_BASE, samp  # noqa: E402
from mappo import MLP, masked_probs  # noqa: E402

N = 160


def trace(actor, setting, seed, cfg):
    ad = S.Adapter(setting, N, seed, CFG['SLOTS'])
    st = S.SLAState(N, dict(S.CHOSEN, **cfg))
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    off, mx, jain = [], [], []
    for t in range(CFG['SLOTS']):
        pred = ad.predicted(t)
        sb = st.proxy(pred)
        m = st.mask(sb, ad.dl)
        a = samp(masked_probs(actor.forward(st.obs(ad, sb, t))[0], m), rg)
        lat, cnt = ad.step(a, t)
        st.update(a, lat, pred, cnt, ad.dl)
        if t >= CFG['WARM']:
            c = cnt[1:].astype(float)
            off.append(c.sum())
            mx.append(c.max())
            jain.append(c.sum() ** 2 / (len(c) * (c ** 2).sum()) if c.sum() else 1.0)
    return np.array(off), np.array(mx), np.array(jain)


def main():
    carry = LOCAL_CARRY or 'idle'
    f = ROOT / 'results' / 'sla_diagnostics.json'
    out = json.load(open(f)) if f.exists() else {}
    for setting in ('homog', 'het50'):
        rows = []
        for path in sorted((ROOT / 'results' / 'sla').glob(f'{setting}_{carry}_N{N}_r*.json')):
            d = json.load(open(path))
            actor = MLP.from_state(d['actor'])
            O, X, J, lag = [], [], [], []
            for s in range(CFG['NSEED']):
                o, x, j = trace(actor, setting, s, d['cfg'])
                O.append(o); X.append(x); J.append(j)
                lag.append(np.corrcoef(o[:-1], o[1:])[0, 1] if o.std() > 0 else 0.0)
            O = np.concatenate(O)
            p = O.mean() / N
            rows.append(dict(run=d['run_seed'], off_per_slot=float(O.mean()), sd=float(O.std()),
                             binomial_sd=float(np.sqrt(N * p * (1 - p))), sd_ratio=float(O.std() / np.sqrt(N * p * (1 - p))),
                             lag1=float(np.nanmean(lag)), max_load=float(np.concatenate(X).mean()),
                             jain=float(np.concatenate(J).mean())))
            print(setting, carry, rows[-1], flush=True)
        if rows:
            out[f'{setting}|{carry}'] = rows
    if carry == 'idle':
        pl = json.load(open(ROOT / 'results' / 'res_planner.json'))['main']['A'][str(N)]['plan_z2']
        out['plan_reference'] = dict(note='planned prior, homog, idle processor, N = 160',
                                     off_per_slot=float(pl['off']) * N, max_load=float(pl['maxl']), jain=float(pl['fair']))
    json.dump(out, open(f, 'w'), indent=1)


if __name__ == '__main__':
    main()
