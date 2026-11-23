"""
exp_marl_value.py  (revision experiment E14, 12 Nov 2026)
=========================================================
Value-based multi-agent DRL baselines for Reviewer 1, comment 4: parameter-
sharing independent DQN (IDQN, individual reward) and value-decomposition
networks (VDN, team reward), in pure NumPy, on the archived gipa_core
environment.

Protocol (mirrors experiments/exp_mappo.py as documented in docs/SCRIPTS.md)
---------------------------------------------------------------------------
* Observation per vehicle = the information GIPA uses: class bit, the four
  intrinsic scores (ms/40), the prior-inclusive deadline mask (4 bits) and
  the heuristic prior c_bar (ms/40).  13 inputs.  No load information.
* One Q-network shared by every vehicle: 13 -> 64 tanh -> 64 tanh -> 4.
  Invalid (masked) actions are never selected.
* Reward per vehicle r_i = -latency_i/10 - 3 * miss_i.
    IDQN : each vehicle regresses Q(o_i, a_i) on its own r_i.
    VDN  : (1/N) sum_i Q(o_i, a_i) is regressed on the fleet mean reward.
* Discount 0 (actions do not change the next state in this simulator, as for
  MAPPO), so no bootstrapped target and no target network.
* Training: 4,000 slots, environments cycling through seed indices
  1000-1049, epsilon-greedy exploration from 1.0 to 0.05 over the first 70%
  of training, replay buffer of 2,000 slots, 32 slots per Adam step
  (lr 1e-3, gradient-norm clip 5), one step per slot after 100 warm-up slots.
  Five training runs, initialisation and exploration seeds 8000 + run.
* Execution rule chosen on the tuning seeds 2000-2009 (never evaluated):
  epsilon-greedy (epsilon 0.02, as the archived Q-learner) or Boltzmann over
  the Q-values with temperature in {0.1, 0.25, 0.5, 1, 2}; the rule with the
  lowest mean latency among those with at most 0.1% class-standardised misses.
* Evaluation: the 30 archived seeds (environment 1000+s, action 5000+s),
  70 slots with 25 warm-up, metrics exactly as gipa_core.run.
Usage:  python -I exp_marl_value.py --src /path/to/src --out res_marl_value.json
"""
import argparse
import json
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--N', type=int, nargs='+', default=[160])
ap.add_argument('--runs', type=int, default=5)
ap.add_argument('--slots', type=int, default=4000)
ap.add_argument('--methods', nargs='+', default=['idqn', 'vdn'])
ap.add_argument('--mask', default='prior', choices=['prior', 'deadline'],
                help='prior: GIPA prior-inclusive deadline mask; deadline: uncongested-latency deadline mask (as MAPPO at N >= 240)')
args = ap.parse_args()
sys.path.insert(0, args.src)
import gipa_core as gc  # noqa: E402

CFG, REG = gc.CFG, gc.REG
R = REG['A']
H = 64
TEMPS = [0.1, 0.25, 0.5, 1.0, 2.0]
EPS_EVAL = 0.02
# Seed indices as in exp_mappo.py (gipa_core adds RNG_ENV_BASE = 1000):
# training 1000-1049, tuning 2000-2009, evaluation 0-29 (the archived seeds).
TRAIN_ENV_BASE, TRAIN_ENV_COUNT = 1000, 50
TUNE_IDX = list(range(2000, 2010))
EVAL_IDX = list(range(30))


def obs_matrix(env, q, cb, N):
    D = env.dl[:N][:, None]
    m = ((-q + (cb[None, :] if args.mask == 'prior' else 0.0)) <= D)
    m[~m.any(1), 0] = True
    x = np.concatenate([env.isS[:N, None].astype(float), q / 40.0,
                        m.astype(float), np.tile(cb / 40.0, (N, 1))], 1)
    return x, m


class QNet:
    def __init__(self, nin, nout, rg):
        s1, s2, s3 = 1 / np.sqrt(nin), 1 / np.sqrt(H), 1 / np.sqrt(H)
        self.p = dict(W1=rg.normal(0, s1, (nin, H)), b1=np.zeros(H),
                      W2=rg.normal(0, s2, (H, H)), b2=np.zeros(H),
                      W3=rg.normal(0, s3, (H, nout)), b3=np.zeros(nout))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        p = self.p
        h1 = np.tanh(x @ p['W1'] + p['b1'])
        h2 = np.tanh(h1 @ p['W2'] + p['b2'])
        return h2 @ p['W3'] + p['b3'], (x, h1, h2)

    def backward(self, cache, dq):
        x, h1, h2 = cache
        p = self.p
        g = {}
        g['W3'] = h2.T @ dq
        g['b3'] = dq.sum(0)
        dh2 = (dq @ p['W3'].T) * (1 - h2 ** 2)
        g['W2'] = h1.T @ dh2
        g['b2'] = dh2.sum(0)
        dh1 = (dh2 @ p['W2'].T) * (1 - h1 ** 2)
        g['W1'] = x.T @ dh1
        g['b1'] = dh1.sum(0)
        return g

    def adam(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        norm = np.sqrt(sum((v ** 2).sum() for v in g.values()))
        if norm > clip:
            g = {k: v * clip / norm for k, v in g.items()}
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)


def reward(env, cost, N):
    return -cost / 10.0 - 3.0 * (cost > env.dl[:N])


def train(method, N, run, slots):
    rg = np.random.default_rng(8000 + run)
    net = QNet(13, CFG['M'] + 1, rg)
    cb = gc.prior_heur(N)
    buf = []
    curve = []
    for st in range(slots):
        eps = max(0.05, 1 - st / (0.7 * slots))
        env = gc.Env(N, TRAIN_ENV_BASE + st % TRAIN_ENV_COUNT, R)
        q = env.q(N)
        x, m = obs_matrix(env, q, cb, N)
        Q, _ = net.forward(x)
        Qm = np.where(m, Q, -np.inf)
        greedy = Qm.argmax(1)
        # uniform random among feasible actions
        u = rg.random((N, 1))
        c = np.cumsum(m / m.sum(1, keepdims=True), 1)
        rnd = np.minimum((u > c).sum(1), CFG['M'])
        a = np.where(rg.random(N) < eps, rnd, greedy)
        cost, _ = env.lat(a, N)
        r = reward(env, cost, N)
        buf.append((x, a, r))
        if len(buf) > 2000:
            buf.pop(0)
        curve.append(float(cost.mean()))
        if st < 100:
            continue
        idx = rg.integers(0, len(buf), 32)
        xs = np.concatenate([buf[i][0] for i in idx])
        as_ = np.concatenate([buf[i][1] for i in idx])
        rs = np.concatenate([buf[i][2] for i in idx])
        Qb, cache = net.forward(xs)
        qsa = Qb[np.arange(len(as_)), as_]
        dq = np.zeros_like(Qb)
        if method == 'idqn':
            err = qsa - rs                       # per-agent regression
            dq[np.arange(len(as_)), as_] = 2 * err / len(as_)
        else:                                    # vdn: per-slot joint regression
            err_slot = []
            for k in range(len(idx)):
                sl = slice(k * N, (k + 1) * N)
                e = qsa[sl].mean() - rs[sl].mean()
                err_slot.append(e)
                dq[np.arange(k * N, (k + 1) * N), as_[sl]] = 2 * e / N / len(idx)
        g = net.backward(cache, dq)
        net.adam(g)
    return net, curve


def policy_actions(net, env, q, cb, rg, N, rule):
    x, m = obs_matrix(env, q, cb, N)
    Q, _ = net.forward(x)
    Qm = np.where(m, Q, -np.inf)
    if rule[0] == 'eps':
        a = Qm.argmax(1)
        f = rg.random(N) < rule[1]
        if f.any():
            u = rg.random((N, 1))
            c = np.cumsum(m / m.sum(1, keepdims=True), 1)
            rnd = np.minimum((u > c).sum(1), CFG['M'])
            a = np.where(f, rnd, a)
        return a
    z = Qm / rule[1]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    return gc.samp(p, rg)


def run_episode(net, N, seed, rule):
    """Same metrics as gipa_core.run."""
    env = gc.Env(N, seed, R)
    rg = np.random.default_rng(gc.RNG_ACT_BASE + seed)
    cb = gc.prior_heur(N)
    out = dict(mS=[], mB=[], lat=[], fair=[], off=[], offS=[], offB=[], en=[], p95=[], maxl=[])
    for t in range(CFG['SLOTS']):
        q = env.q(N)
        a = policy_actions(net, env, q, cb, rg, N, rule)
        cost, c = env.lat(a, N)
        e = env.en(a, N)
        if t >= CFG['WARM']:
            S = env.isS[:N]
            miss = cost > env.dl[:N]
            out['mS'].append(miss[S].mean() if S.any() else 0.)
            out['mB'].append(miss[~S].mean() if (~S).any() else 0.)
            out['lat'].append(cost.mean())
            out['p95'].append(np.percentile(cost, 95))
            rl = c[1:]
            out['fair'].append(np.nan if rl.sum() == 0 else rl.sum() ** 2 / (len(rl) * (rl ** 2).sum()))
            out['off'].append((a > 0).mean())
            out['offS'].append((a[S] > 0).mean() if S.any() else 0.)
            out['offB'].append((a[~S] > 0).mean() if (~S).any() else 0.)
            out['en'].append(e.mean())
            out['maxl'].append(rl.max())
    return {k: gc.safe_mean(v) for k, v in out.items()}


def mean_of(rs):
    return {k: gc.safe_mean([r[k] for r in rs]) for k in rs[0]}


def main():
    res = dict(protocol=__doc__, mask=args.mask, methods={})
    rules = [('eps', EPS_EVAL)] + [('boltz', T) for T in TEMPS]
    for method in args.methods:
        res['methods'][method] = {}
        for N in args.N:
            t0 = time.time()
            runs_out = []
            for run in range(args.runs):
                net, curve = train(method, N, run, args.slots)
                # tuning: choose the execution rule on tuning seeds
                tune = {}
                for rule in rules:
                    rs = [run_episode(net, N, s, rule) for s in TUNE_IDX]
                    mm = mean_of(rs)
                    tune[f'{rule[0]}:{rule[1]}'] = mm
                ok = {k: v for k, v in tune.items() if 0.3 * v['mS'] + 0.7 * v['mB'] <= 0.001}
                pool = ok if ok else tune
                best = min(pool, key=lambda k: pool[k]['lat'])
                rule = ('eps', EPS_EVAL) if best.startswith('eps') else ('boltz', float(best.split(':')[1]))
                # greedy policy per class (diagnostic)
                env = gc.Env(N, 0, R)
                x, m = obs_matrix(env, env.q(N), gc.prior_heur(N), N)
                Q, _ = net.forward(x)
                iS = int(np.argmax(env.isS[:N]))
                iB = int(np.argmax(~env.isS[:N]))
                ev = [run_episode(net, N, s, rule) for s in EVAL_IDX]
                ev_eps = [run_episode(net, N, s, ('eps', EPS_EVAL)) for s in EVAL_IDX]
                runs_out.append(dict(run=run, rule=best, tune=tune, eval=mean_of(ev), eval_seeds=ev,
                                     eval_eps_greedy=mean_of(ev_eps),
                                     Q_safety=Q[iS].tolist(), Q_background=Q[iB].tolist(),
                                     curve_last500=float(np.mean(curve[-500:])),
                                     curve=[float(np.mean(curve[i:i + 100])) for i in range(0, len(curve), 100)]))
                print(f'{method} N={N} run {run}: rule={best} lat={runs_out[-1]["eval"]["lat"]:.2f} '
                      f'mS={runs_out[-1]["eval"]["mS"]*100:.2f}% mB={runs_out[-1]["eval"]["mB"]*100:.2f}% '
                      f'off={runs_out[-1]["eval"]["off"]:.3f}  [eps-greedy: {runs_out[-1]["eval_eps_greedy"]["lat"]:.2f} ms]  '
                      f'QS={np.round(Q[iS],3)} QB={np.round(Q[iB],3)}', flush=True)
            summ = mean_of([r['eval'] for r in runs_out])
            summ_eps = mean_of([r['eval_eps_greedy'] for r in runs_out])
            lat_runs = [r['eval']['lat'] for r in runs_out]
            res['methods'][method][str(N)] = dict(runs=runs_out, mean=summ, mean_eps_greedy=summ_eps,
                                                  lat_min=min(lat_runs), lat_max=max(lat_runs),
                                                  seconds=time.time() - t0)
            print(f'== {method} N={N}: mean lat {summ["lat"]:.2f} (runs {min(lat_runs):.2f}-{max(lat_runs):.2f}) '
                  f'mS {summ["mS"]*100:.2f}% mB {summ["mB"]*100:.2f}% off {summ["off"]:.3f} offS {summ["offS"]:.3f} '
                  f'offB {summ["offB"]:.3f} p95 {summ["p95"]:.2f} fair {summ["fair"]:.3f}  ({time.time()-t0:.0f}s)', flush=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == '__main__':
    main()
