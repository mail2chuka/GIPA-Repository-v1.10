import sys; sys.path.insert(0, sys.argv[1])
import numpy as np
from gipa_core import REG, CFG, prior_heur
from capacity import run_ac, capacities
from planner import plan_allocation
R = REG['B']
print('capacities', capacities(R))
for N in (80, 160, 240):
    for name, kw in (('plan z=0', dict(z=0.0)), ('plan z=2', dict(z=2.0))):
        try:
            plan = plan_allocation(N, R, **kw)
        except Exception as e:
            print(N, name, 'infeasible', e); continue
        if plan is None or (isinstance(plan, dict) and plan.get('feasible') is False):
            print(N, name, 'infeasible'); continue
        rs = [run_ac('plan', N, s, R, ac=False, plan=plan) for s in range(CFG['NSEED'])]
        m = {k: float(np.nanmean([r[k] for r in rs])) for k in ('lat','mS','mB','over_sel','sev_sel','maxl_sel','off','offS','offB')}
        print(N, name, {k: round(v,4) for k,v in m.items()})
    rs = [run_ac('gipa', N, s, R, ac=False) for s in range(CFG['NSEED'])]
    m = {k: float(np.nanmean([r[k] for r in rs])) for k in ('lat','mS','mB','over_sel','sev_sel','maxl_sel','off','offS','offB')}
    print(N, 'gipa heuristic', {k: round(v,4) for k,v in m.items()})
