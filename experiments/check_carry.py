"""
check_carry.py
==============
Experiment E13 check: the local-queue chain of ``src/carry.py`` against a
direct simulation of the backlog recursion used by the simulators.

For one vehicle with offload (sampling) probability f, local service L and
deadline D, the simulators apply, slot by slot,
    W' = max(W + L 1{local} - 20, 0),
where in 'own' mode a task with W + L > D is sent to an RSU instead of run
locally. ``carry.local_chain`` gives the stationary share of local tasks,
their mean latency W + L and their miss rate. This script simulates the
recursion for 2,000,000 slots per case and prints both.

Cases: Regime A background (L 22, D 40) and Regime B background (L 30, D 40),
blind and own, at several f.

Output
------
results/carry_check.json: list of {L, D, own, f, chain, sim}.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python experiments/check_carry.py
"""

import json
from pathlib import Path

import numpy as np

from carry import local_chain

ROOT = Path(__file__).resolve().parents[1]
SLOTS, BURN, SEED = 2_000_000, 10_000, 20261108


def simulate(f, L, D, own, slot=20.0):
    rg = np.random.default_rng(SEED)
    off = rg.random(SLOTS) < f
    W, n_loc, lat_sum, miss = 0.0, 0, 0.0, 0
    for t in range(SLOTS):
        local = not off[t]
        if local and own and W + L > D:
            local = False
        if local and t >= BURN:
            n_loc += 1
            lat_sum += W + L
            miss += (W + L > D)
        W = max(W + (L if local else 0.0) - slot, 0.0)
    n = SLOTS - BURN
    return dict(share_local=n_loc / n, lat_local=lat_sum / n_loc if n_loc else L,
                miss_local=miss / n_loc if n_loc else 0.0)


def main():
    out = []
    for L, D in [(22.0, 40.0), (30.0, 40.0)]:
        for own in [False, True]:
            for f in [0.0, 0.1, 0.2, 0.4]:
                ch = local_chain(f, L, D, own)
                if not ch['stable']:
                    print(f'L {L:.0f} own {own!s:5} f {f:.1f}: chain unstable (simulation skipped)', flush=True)
                    out.append(dict(L=L, D=D, own=own, f=f, chain=dict(stable=False), sim=None))
                    continue
                sm = simulate(f, L, D, own)
                print(f"L {L:.0f} own {own!s:5} f {f:.1f}: share {ch['share_local']:.4f}/{sm['share_local']:.4f} "
                      f"lat {ch['lat_local']:.3f}/{sm['lat_local']:.3f} miss {ch['miss_local']:.4f}/{sm['miss_local']:.4f}",
                      flush=True)
                out.append(dict(L=L, D=D, own=own, f=f,
                                chain={k: ch[k] for k in ['share_local', 'lat_local', 'miss_local', 'stable']}, sim=sm))
    with open(ROOT / 'results' / 'carry_check.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', ROOT / 'results' / 'carry_check.json')


if __name__ == '__main__':
    main()
