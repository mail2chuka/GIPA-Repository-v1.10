"""
mappo.py
========
Multi-agent PPO (MAPPO) baseline for workplan experiment E7, answering
Reviewer 1 comment 4 ("compare your work with novel multi-agent DRL
methods").

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

Implementation
--------------
Pure NumPy, so it runs on any CPU with only the packages in requirements.txt
(no PyTorch). Networks are small multilayer perceptrons (2 hidden layers of
64 tanh units) trained with Adam; gradients are written by hand and verified
against finite differences (:func:`gradient_check`).

Algorithm (centralised training, decentralised execution)
---------------------------------------------------------
* Parameter sharing: one actor network is shared by all vehicles.
* Actor input (per vehicle, the same information GIPA uses): class one-hot,
  intrinsic action scores, prior-inclusive deadline mask and the congestion
  prior. Variant 'load' adds the previous slot's RSU load shares (more
  information than GIPA has).
* Invalid actions are removed by masking the logits. ``mask='prior'``
  (default) is the same prior-inclusive deadline mask GIPA uses, with the
  local fallback, so MAPPO never selects an action GIPA considers
  infeasible. At N >= 240 that mask forbids every RSU (the heuristic prior
  alone exceeds the background deadline margin), so MAPPO can then only
  stay local. ``mask='deadline'`` checks deadlines against the uncongested
  latency only (no prior), which in the archived regimes allows every RSU;
  congestion is then learned from the miss penalty. Use it for a fair
  comparison at high density.
* Critic: 'central' (MAPPO) sees the vehicle's observation plus global
  features available only during training (previous-slot RSU load shares,
  safety fraction, fleet size); 'local' (IPPO fallback) sees only the
  vehicle's observation.
* Reward per vehicle and slot: r_i = -latency_i / 10 - pen * miss_i.
  Modes: 'individual' (r_i), 'team' (fleet mean of r_i, shared by every
  vehicle) and 'mixed' (the average of the two).
* PPO clipped surrogate (clip 0.2), entropy bonus, separate critic
  optimiser, advantage normalisation, global gradient-norm clipping (0.5).
* Discount: gamma = 0 by default. In this simulator the next state does not
  depend on the actions (classes and latencies are fixed per episode), so
  the optimal policy is myopic and gamma > 0 only adds variance. The 'load'
  variant's observations do depend on past actions; use gamma = 0.9 with
  GAE lambda = 0.95 for it (``train(..., gamma=0.9)``).

Seeds (disjoint from evaluation)
--------------------------------
* Training environments: seed indices 1000-1049 (environment RNG 2000-2049),
  cycled.
* Training action sampling and network initialisation: 8000 + run seed.
* Tuning evaluation: seed indices 2000-2009 (environment RNG 3000-3009).
* Final evaluation: the 30 archived seeds (environment 1000-1029, action
  sampling 5000-5029), so MAPPO results pair with every other policy.
  Evaluation samples actions from the learned stochastic policy; a greedy
  argmax would herd.

Latency models
--------------
'affine' : gipa_core.Env (the submitted model)
'ps'     : calibration.PSEnv (Eq. 5 with class-dependent compute, E2)
"""

import time

import numpy as np

from gipa_core import CFG, Env, samp, safe_mean, prior_heur, RNG_ACT_BASE

TRAIN_SEED_INDEX = 1000        # training environments use seed indices 1000..1049
N_TRAIN_ENVS = 50
TUNE_SEED_INDEX = 2000         # tuning evaluation uses seed indices 2000..2009
RNG_TRAIN_BASE = 8000


# ----------------------------------------------------------------------
# Networks and optimiser
# ----------------------------------------------------------------------
class MLP:
    """Fully connected network with tanh hidden layers and a linear output.

    ``forward`` returns the output and the list of layer activations needed
    by ``backward``. Parameters are float64.
    """

    def __init__(self, sizes, rng, out_scale=0.01):
        self.W, self.b = [], []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            scale = out_scale if last else 1.0 / np.sqrt(n_in)
            self.W.append(rng.standard_normal((n_in, n_out)) * scale)
            self.b.append(np.zeros(n_out))

    def params(self):
        """All parameter arrays (updated in place by the optimiser)."""
        return self.W + self.b

    def forward(self, X):
        """Return (output, activations)."""
        hs = [X]
        h = X
        for i in range(len(self.W)):
            z = h @ self.W[i] + self.b[i]
            h = np.tanh(z) if i < len(self.W) - 1 else z
            hs.append(h)
        return h, hs

    def backward(self, hs, dout):
        """Gradients of a scalar loss given dLoss/dOutput; same order as params()."""
        n = len(self.W)
        gW, gb = [None] * n, [None] * n
        d = dout
        for i in reversed(range(n)):
            gW[i] = hs[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (1.0 - hs[i] ** 2)
        return gW + gb

    def state(self):
        """Parameters as plain lists (for JSON)."""
        return {'W': [w.tolist() for w in self.W], 'b': [v.tolist() for v in self.b]}

    @classmethod
    def from_state(cls, st):
        """Rebuild a network from :meth:`state`."""
        net = cls.__new__(cls)
        net.W = [np.array(w) for w in st['W']]
        net.b = [np.array(v) for v in st['b']]
        return net


class Adam:
    """Adam optimiser with global gradient-norm clipping."""

    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8, clip_norm=0.5):
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip_norm
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        """One update; ``params`` are modified in place."""
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if self.clip and norm > self.clip:
            grads = [g * (self.clip / norm) for g in grads]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------
# Masked categorical policy and PPO loss
# ----------------------------------------------------------------------
def masked_probs(logits, mask):
    """Softmax over the allowed actions only; disallowed actions get exactly 0."""
    z = np.where(mask, logits, -1e9)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z) * mask
    return e / e.sum(1, keepdims=True)


def ppo_actor_loss(logits, mask, a, logp_old, adv, clip=0.2, ent_coef=0.01):
    """PPO clipped-surrogate loss with entropy bonus, and its gradient.

    Returns (loss, dLoss/dlogits, diagnostics). The loss is
    -mean(min(rho A, clip(rho) A)) - ent_coef * mean(H), with rho the
    probability ratio and H the entropy over allowed actions.
    """
    B = len(a)
    p = masked_probs(logits, mask)
    logp_all = np.where(mask, np.log(np.where(mask, p, 1.0)), 0.0)
    rows = np.arange(B)
    logp = logp_all[rows, a]
    ratio = np.exp(logp - logp_old)
    s1 = ratio * adv
    s2 = np.clip(ratio, 1 - clip, 1 + clip) * adv
    H = -(p * logp_all).sum(1)
    loss = -np.minimum(s1, s2).mean() - ent_coef * H.mean()
    use = (s1 <= s2).astype(float)          # gradient flows only through the unclipped term
    onehot = np.zeros_like(p)
    onehot[rows, a] = 1.0
    d = ((-adv * use / B) * ratio)[:, None] * (onehot - p)
    d += (ent_coef / B) * p * (logp_all + H[:, None])
    d = np.where(mask, d, 0.0)
    diag = dict(entropy=float(H.mean()), clip_frac=float((np.abs(ratio - 1) > clip).mean()),
                approx_kl=float((logp_old - logp).mean()))
    return float(loss), d, diag


def gradient_check(seed=0, eps=1e-6):
    """Compare hand-written actor and critic gradients with finite differences.

    Returns the largest relative error over a random sample of parameters
    (should be below about 1e-6).
    """
    rng = np.random.default_rng(seed)
    B, d_in, n_a = 12, 7, 4
    X = rng.standard_normal((B, d_in))
    mask = rng.random((B, n_a)) < 0.75
    mask[:, 0] = True
    net = MLP([d_in, 8, 8, n_a], rng, out_scale=0.5)
    logits, _ = net.forward(X)
    p = masked_probs(logits, mask)
    a = np.array([rng.choice(np.flatnonzero(m), p=pr[m] / pr[m].sum()) for m, pr in zip(mask, p)])
    logp_old = np.log(p[np.arange(B), a]) + rng.normal(0, 0.05, B)   # ratios near 1, clear of kinks
    adv = rng.standard_normal(B)

    def actor_loss():
        lg, _ = net.forward(X)
        return ppo_actor_loss(lg, mask, a, logp_old, adv)[0]

    lg, hs = net.forward(X)
    _, dl, _ = ppo_actor_loss(lg, mask, a, logp_old, adv)
    grads = net.backward(hs, dl)
    worst = 0.0
    for P, G in zip(net.params(), grads):
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + eps
            lp = actor_loss()
            P[idx] = old - eps
            lm = actor_loss()
            P[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - G[idx]) / max(1e-8, abs(num) + abs(G[idx])))
    # critic: 0.5 * mean squared error
    cnet = MLP([d_in, 8, 8, 1], rng, out_scale=0.5)
    y = rng.standard_normal(B)

    def critic_loss():
        v, _ = cnet.forward(X)
        return 0.5 * ((v[:, 0] - y) ** 2).mean()

    v, hs = cnet.forward(X)
    grads = cnet.backward(hs, ((v[:, 0] - y) / B)[:, None])
    for P, G in zip(cnet.params(), grads):
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + eps
            lp = critic_loss()
            P[idx] = old - eps
            lm = critic_loss()
            P[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - G[idx]) / max(1e-8, abs(num) + abs(G[idx])))
    return worst


# ----------------------------------------------------------------------
# Environment adapter and observations
# ----------------------------------------------------------------------
def make_env(N, seed, R, model):
    """Environment for seed index ``seed``: 'affine' (gipa_core.Env) or 'ps' (Eq. 5)."""
    if model == 'affine':
        return Env(N, seed, R)
    if model == 'ps':
        from calibration import PSEnv, make_calibration
        return PSEnv(N, seed, R, make_calibration())
    raise ValueError(model)


def feasibility_mask(env, q, cb, N, kind='prior'):
    """Deadline mask with the local fallback.

    kind 'prior': predicted latency = intrinsic cost + prior (GIPA's mask).
    kind 'deadline': intrinsic (uncongested) cost only, no prior.
    """
    if kind == 'deadline':
        cb = np.zeros_like(cb)
    elif kind != 'prior':
        raise ValueError(kind)
    m = (-q + cb[None, :]) <= env.dl[:N][:, None]
    m[~m.any(1), 0] = True
    return m


def build_obs(env, q, cb, mask, prev_share, N, variant):
    """Actor observation per vehicle (see module docstring)."""
    M1 = CFG['M'] + 1
    S = env.isS[:N]
    parts = [np.stack([S, ~S], 1).astype(float), q / 40.0, mask.astype(float),
             np.broadcast_to(cb / 50.0, (N, M1))]
    if variant == 'load':
        parts.append(np.broadcast_to(prev_share, (N, M1)))
    return np.concatenate(parts, 1)


def build_critic_in(obs, env, prev_share, N, critic):
    """Critic input: the actor observation plus global features ('central') or alone ('local')."""
    if critic == 'local':
        return obs
    glob = np.concatenate([prev_share, [env.isS[:N].mean(), N / 320.0]])
    return np.concatenate([obs, np.broadcast_to(glob, (N, len(glob)))], 1)


def obs_dims(variant, critic):
    """(actor input size, critic input size)."""
    M1 = CFG['M'] + 1
    d_actor = 2 + 3 * M1 + (M1 if variant == 'load' else 0)
    d_critic = d_actor + (0 if critic == 'local' else M1 + 2)
    return d_actor, d_critic


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
DEFAULTS = dict(reward='team', lr=1e-3, critic_lr=1e-3, ent_coef=0.01, pen=3.0,
                clip=0.2, epochs=4, minibatch=4096, gamma=0.0, gae_lambda=0.95,
                hidden=64, variant='base', critic='central', model='affine', mask='prior')


def train(N, R, run_seed, iters=400, log_every=0, **cfg):
    """Train one MAPPO policy at fleet size N.

    Each iteration rolls out one full episode (70 slots, all N vehicles) on
    a training environment and performs ``epochs`` PPO passes over its
    N x 70 samples. Returns a dict with the trained networks, the per-
    iteration learning curve, a plateau verdict and the wall-clock time.
    """
    c = dict(DEFAULTS, **cfg)
    rng = np.random.default_rng(RNG_TRAIN_BASE + run_seed)
    M1 = CFG['M'] + 1
    d_a, d_c = obs_dims(c['variant'], c['critic'])
    actor = MLP([d_a, c['hidden'], c['hidden'], M1], rng, out_scale=0.01)
    critic = MLP([d_c, c['hidden'], c['hidden'], 1], rng, out_scale=1.0)
    opt_a = Adam(actor.params(), c['lr'])
    opt_c = Adam(critic.params(), c['critic_lr'])
    cb = prior_heur(N)
    T = CFG['SLOTS']
    curve = []
    t0 = time.time()

    for it in range(iters):
        env = make_env(N, TRAIN_SEED_INDEX + (it + run_seed * 7) % N_TRAIN_ENVS, R, c['model'])
        q = env.q(N)
        mask = feasibility_mask(env, q, cb, N, c['mask'])
        prev = np.full(M1, 1.0 / M1)
        OBS, CIN, ACT, LOGP, REW, VAL = [], [], [], [], [], []
        lat_sum, missS, missB = 0.0, [], []
        S = env.isS[:N]
        for t in range(T):
            obs = build_obs(env, q, cb, mask, prev, N, c['variant'])
            cin = build_critic_in(obs, env, prev, N, c['critic'])
            logits, _ = actor.forward(obs)
            p = masked_probs(logits, mask)
            a = samp(p, rng)
            lat, cnt = env.lat(a, N)
            miss = lat > env.dl[:N]
            r = -lat / 10.0 - c['pen'] * miss
            if c['reward'] == 'team':
                r = np.full(N, r.mean())
            elif c['reward'] == 'mixed':
                r = 0.5 * r + 0.5 * r.mean()
            v, _ = critic.forward(cin)
            OBS.append(obs); CIN.append(cin); ACT.append(a)
            LOGP.append(np.log(p[np.arange(N), a])); REW.append(r); VAL.append(v[:, 0])
            lat_sum += lat.mean()
            missS.append(miss[S].mean() if S.any() else 0.0)
            missB.append(miss[~S].mean() if (~S).any() else 0.0)
            prev = cnt / N

        REW, VAL = np.array(REW), np.array(VAL)              # (T, N)
        adv = np.zeros_like(REW)
        last = np.zeros(N)
        for t in reversed(range(T)):
            nextv = VAL[t + 1] if t < T - 1 else 0.0
            delta = REW[t] + c['gamma'] * nextv - VAL[t]
            last = delta + c['gamma'] * c['gae_lambda'] * last
            adv[t] = last
        ret = (adv + VAL).ravel()
        adv = adv.ravel()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        X = np.concatenate(OBS)
        XC = np.concatenate(CIN)
        A = np.concatenate(ACT)
        LP = np.concatenate(LOGP)
        MK = np.tile(mask, (T, 1))
        Btot = len(A)
        diag = {}
        for _ in range(c['epochs']):
            perm = rng.permutation(Btot)
            for s in range(0, Btot, c['minibatch']):
                idx = perm[s:s + c['minibatch']]
                lg, hs = actor.forward(X[idx])
                _, dl, diag = ppo_actor_loss(lg, MK[idx], A[idx], LP[idx], adv[idx],
                                             c['clip'], c['ent_coef'])
                opt_a.step(actor.params(), actor.backward(hs, dl))
                v, hs = critic.forward(XC[idx])
                dv = ((v[:, 0] - ret[idx]) / len(idx))[:, None]
                opt_c.step(critic.params(), critic.backward(hs, dv))
        row = dict(iter=it, mean_reward=float(REW.mean()), lat=lat_sum / T,
                   mS=float(np.mean(missS)), mB=float(np.mean(missB)),
                   off=float((np.concatenate(ACT) > 0).mean()), **diag)
        curve.append(row)
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"  it {it:4d}  reward {row['mean_reward']:7.3f}  lat {row['lat']:6.2f}  "
                  f"miss S/B {100 * row['mS']:5.1f}/{100 * row['mB']:4.1f}%  off {row['off']:.2f}  "
                  f"H {row.get('entropy', 0):.3f}", flush=True)

    k = max(1, iters // 10)
    last = np.mean([r['mean_reward'] for r in curve[-k:]])
    before = np.mean([r['mean_reward'] for r in curve[-2 * k:-k]]) if iters >= 2 * k else last
    plateau = abs(last - before) <= 0.01 * abs(before)
    return dict(actor=actor, critic=critic, curve=curve, config=c, N_train=N, run_seed=run_seed,
                plateau=bool(plateau), plateau_rel_change=float(abs(last - before) / max(abs(before), 1e-12)),
                seconds=time.time() - t0, seconds_per_iter=(time.time() - t0) / max(iters, 1))


# ----------------------------------------------------------------------
# Evaluation (same metrics, seeds and draw order as gipa_core.run)
# ----------------------------------------------------------------------
def run_mappo(actor, N, seed, R, variant='base', model='affine', mask='prior'):
    """One evaluation episode on archived seed index ``seed``.

    Actions are sampled from the learned stochastic policy with the archived
    action generator (5000 + seed) via ``gipa_core.samp``, so results pair
    with every other policy. ``mask`` must match the one used in training.
    Returns the metric dict of ``gipa_core.run``.
    """
    env = make_env(N, seed, R, model)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    cb = prior_heur(N)
    M1 = CFG['M'] + 1
    prev = np.full(M1, 1.0 / M1)
    out = dict(mS=[], mB=[], lat=[], fair=[], off=[], offS=[], offB=[], en=[], p95=[], maxl=[])
    for t in range(CFG['SLOTS']):
        q = env.q(N)
        m_t = feasibility_mask(env, q, cb, N, mask)
        obs = build_obs(env, q, cb, m_t, prev, N, variant)
        logits, _ = actor.forward(obs)
        a = samp(masked_probs(logits, m_t), rg)
        cost, c = env.lat(a, N)
        e = env.en(a, N)
        prev = c / N
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
    return {k: safe_mean(v) for k, v in out.items()}


def evaluate(actor, N, R, seeds=None, variant='base', model='affine', mask='prior'):
    """Evaluate on seed indices (default: the 30 archived seeds 0..29).

    Returns (across-seed means, per-seed list).
    """
    seeds = range(CFG['NSEED']) if seeds is None else seeds
    rs = [run_mappo(actor, N, s, R, variant, model, mask) for s in seeds]
    return {k: safe_mean([r[k] for r in rs]) for k in rs[0]}, rs
