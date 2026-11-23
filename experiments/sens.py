"""E15 sensitivity: the re-solved plan with a relaxed fallback in slots without a feasible plan (run with TAN_RELAX=1 GIPA_LOCAL_CARRY=own from the repository root)."""
import sys, json, time
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import exp_tan as X

def job(a):
    arm, sc, N, s = a
    return a, X.run_one(arm, sc, N, s, 0.0, 0.4)

if __name__ == '__main__':
    X.init(os.path.abspath('src'))
    jobs = [(arm, sc, N, s) for sc in ('het50', 'het50cov2', 'het50cov2p20') for N in (160, 240, 320)
            for arm in ('tan', 'tan_z2') for s in range(30)]
    t0 = time.time()
    with Pool(2, initializer=X.init, initargs=(os.path.abspath('src'),)) as p:
        res = p.map(job, jobs, chunksize=3)
    out = {}
    for (arm, sc, N, s), r in res:
        out.setdefault(f'{sc}|{N}|{arm}', []).append(r)
    json.dump(out, open('results/res_tan_carry_relax.json', 'w'))
    print('done', round(time.time() - t0), 's')
