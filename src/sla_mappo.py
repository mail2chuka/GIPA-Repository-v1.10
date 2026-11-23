"""
sla_mappo.py
============
Experiment E16 (November 2026): a recent task-specific vehicular DRL
baseline, SLA-MA-PPO of Georgiades et al., "Enhanced vehicular edge
offloading through SLA-based deep reinforcement learning and risk-aware
bandits", IEEE Trans. Netw. Service Manag., 2026 (manuscript reference
[24]), adapted to the slotted environments of this repository.

Answers Reviewer 1 comment 4 with a method designed for vehicular
offloading rather than a generic multi-agent learner.

What is taken from the paper (Section V and Table III of the accepted
manuscript, City Research Online eprint 38473)
-----------------------------------------------------------------------
* Multi-agent PPO with one actor and one critic shared by all vehicles,
  actor MLP 12 -> 64 -> 64 -> |A|, critic MLP 12 -> 64 -> 64 -> 1.
* Clipped PPO with GAE: learning rate 3e-3, gamma 0.99, lambda 0.95,
  clip 0.20, 4 epochs, minibatch 32.
* SLA admissibility mask (Eqs. 27-28): an action is masked when its
  full-service proxy sb(a) exceeds the margin-tightened deadline
  rho_tau * D (rho_tau = 0.95) or its risk exceeds eta = 0.16; if every
  action is masked the unmasked policy is used.
* Full-service proxy: measured path component (here the uncongested cost
  the vehicle predicts) plus a learned residual updated only from the
  vehicle's own outcomes on the selected action (restless, per arm).
* Reward r = -L_local - L_glob - beta_risk R(a) + B_early with
    L_local = DelayRatio + beta_tau (1{S > rho_tau D} + 1{S > D}),
    L_glob  = w * ([FailPct - FailPct*]+ + [VMUtil - VMUtil*]+ + p99Over),
    B_early = gamma_early (0.5 - l_t)+.
  [24] also offers a projected dual update of w; here it diverged (the
  multiplier exceeded 400 after 800 iterations because misses in the warm-up
  slots keep FailPct above its target), so the fixed-weight form is used
  (w = 1; CHOSEN['dual'] = True restores the dual update).
* Shared KPIs (FailPct, VMUtil, p99Over) and the load proxy l_t (normalised
  active-vehicle count) are broadcast to every vehicle and enter its state.

What had to be chosen (not given in the paper; stated in Section S14)
---------------------------------------------------------------------
* Actions: local execution and RSU 1..M (the paper offers edge and two
  cloud paths and no local execution; here every policy shares GIPA's
  action space). Energy is not in this model, so beta_E = 0.
* The 12 state features (the paper gives only the dimension):
    task size (local service time / 40 ms), class one-hot (2),
    sb(a) / D for the M + 1 actions (clipped at 3), FailPct, VMUtil,
    p99Over of the previous slot, l_t = N / 320, own local backlog / 40 ms.
* DelayRatio = S / local service time (the paper divides by task length).
* VMUtil = mean RSU load / 17 tasks (the Regime A safety cap);
  p99Over = min(max(q99(S / D) - 1, 0), 1) over the previous slot;
  FailPct* = 0.01, VMUtil* = 0.8.
* Risk R(a) = the vehicle's exponentially weighted miss rate on action a
  (weight 0.1); residuals use the same weight.
* The learning rate (3e-3 of [24] or 3e-4) and beta_tau are tuned on
  separate seeds (exp_sla.py tune), beta_risk = 0.5; gamma_early = 0.1,
  entropy 0.01, separate critic optimiser, gradient-norm clip 0.5, rewards
  scaled by the running standard deviation of discounted returns.
* Model selection: from the middle of training the actor is evaluated every
  25 iterations on the tuning seeds and the best checkpoint is kept (see
  train), because the learning curve oscillates.
* Each PPO update uses one 70-slot episode of every vehicle (the paper's
  256-transition buffer is a single orchestrator's buffer); GAE runs along
  each vehicle's own trajectory.

Seeds as in mappo.py: training environments 1000-1049, tuning 2000-2009,
evaluation the 30 archived seeds 0-29 with the archived action generator
5000 + seed, so results pair with every other policy.

Feedback timing (GIPA_SLA_FEEDBACK)
-----------------------------------
unset or 'ideal'  (E16) every latency and deadline outcome of slot t, and the
                  KPIs computed from them, are known at the decision of slot
                  t + 1, including outcomes of tasks still running then.
'gated'           (E19) an outcome is used only once it physically exists at
                  decision time 20 t' ms: a latency (residual update, KPIs) at
                  the task's completion, 20 t + lat; a deadline outcome (risk
                  update) at completion or, for a late task, when its deadline
                  passes, 20 t + min(lat, D). FailPct and p99Over are computed
                  over the outcomes completed during the previous slot (kept
                  from the slot before if none completed); VMUtil uses the RSU
                  association counts of the previous slot, which the RSUs
                  know at submission. Events are applied in time order.

Settings
--------
'homog' : gipa_core.Env, Regime A, affine latency (the main experiment)
'het50' : hetero.HetEnv with the het50 scenario of E12
With GIPA_LOCAL_CARRY=own each is run with carried-over local work.
"""

import os
import time

import numpy as np

from gipa_core import CFG, REG, Env, samp, safe_mean, RNG_ACT_BASE, LOCAL_CARRY, carry_backlog, SLOT_MS
from mappo import MLP, Adam, masked_probs, ppo_actor_loss

TRAIN_SEED_INDEX = 1000
N_TRAIN_ENVS = 50
TUNE_SEEDS = range(2000, 2010)
RNG_TRAIN_BASE = 8600
SAFETY_CAP = 17.0
HET50 = dict(sigma_h=0.5, sigma_r=0.3, coverage=3)

PAPER = dict(lr=3e-3, gamma=0.99, gae_lambda=0.95, clip=0.2, epochs=4, minibatch=32, hidden=64,
             rho_tau=0.95, eta=0.16)
CHOSEN = dict(beta_tau=1.0, beta_risk=0.5, w_glob=1.0, dual=False, eta_lambda=0.01, gamma_early=0.1, fail_star=0.01,
              util_star=0.8, ewma=0.1, ent_coef=0.01, critic_lr=3e-3)
N_OBS = 12
FEEDBACK = os.environ.get('GIPA_SLA_FEEDBACK', 'ideal').strip().lower() or 'ideal'
if FEEDBACK not in ('ideal', 'gated'):
    raise ValueError(f'GIPA_SLA_FEEDBACK must be ideal or gated, not {FEEDBACK!r}')


class Adapter:
    """Common interface to gipa_core.Env ('homog') and hetero.HetEnv ('het50')."""

    def __init__(self, setting, N, seed, slots):
        self.N = N
        self.setting = setting
        if setting == 'homog':
            self.env = Env(N, seed, REG['A'])
        elif setting == 'het50':
            from hetero import HetEnv
            self.env = HetEnv(N, seed, REG['A'], HET50['sigma_h'], HET50['sigma_r'], HET50['coverage'], 0.0,
                              slots=slots)
        else:
            raise ValueError(setting)
        self.isS = self.env.isS[:N]
        self.dl = self.env.dl[:N]

    def predicted(self, t):
        """Uncongested cost the vehicle predicts per action (N, M + 1); includes its own backlog in own mode."""
        if self.setting == 'homog':
            return -self.env.q(self.N)
        return self.env.cost(t, predicted=True)

    def size(self, t):
        """Local service time of this slot's task (ms), excluding any wait."""
        if self.setting == 'homog':
            return self.env.loc[:self.N]
        from hetero import F_LOCAL
        return self.env.c_hat[t] / F_LOCAL

    def backlog(self):
        return carry_backlog(self.env, self.N) if LOCAL_CARRY else np.zeros(self.N)

    def step(self, a, t):
        if self.setting == 'homog':
            return self.env.lat(a, self.N)
        return self.env.lat(a, t)


class SLAState:
    """Per-vehicle learned residuals and risks, and the broadcast KPIs."""

    def __init__(self, N, cfg):
        M1 = CFG['M'] + 1
        self.c = cfg
        self.resid = np.zeros((N, M1))
        self.risk = np.zeros((N, M1))
        self.kpi = np.zeros(3)          # FailPct, VMUtil, p99Over of the previous slot

    def proxy(self, pred):
        return pred + self.resid

    def mask(self, sb, dl):
        ok = np.isfinite(sb)
        m = ok & (sb <= PAPER['rho_tau'] * dl[:, None]) & (self.risk <= PAPER['eta'])
        none = ~m.any(1)
        m[none] = ok[none]                  # every action screened out: unmasked policy (reachable actions)
        return m

    def obs(self, ad, sb, t):
        N = ad.N
        S = ad.isS
        ratio = np.clip(np.where(np.isfinite(sb), sb, 3 * ad.dl[:, None]) / ad.dl[:, None], 0, 3)
        return np.concatenate([(ad.size(t) / 40.0)[:, None], np.stack([S, ~S], 1).astype(float), ratio,
                               np.broadcast_to(self.kpi, (N, 3)), np.full((N, 1), N / 320.0),
                               (ad.backlog() / 40.0)[:, None]], 1)

    def update(self, a, lat, pred, cnt, dl):
        N = len(a)
        rows = np.arange(N)
        w = self.c['ewma']
        miss = lat > dl
        off = a > 0
        r_obs = lat - pred[rows, a]
        self.resid[rows[off], a[off]] = (1 - w) * self.resid[rows[off], a[off]] + w * r_obs[off]
        self.risk[rows, a] = (1 - w) * self.risk[rows, a] + w * miss
        util = cnt[1:].mean() / SAFETY_CAP
        p99 = np.percentile(lat / dl, 99)
        self.kpi = np.array([miss.mean(), util, min(max(p99 - 1.0, 0.0), 1.0)])
        return miss

    def advance(self, t_next):
        """Apply the outcomes known at the decision of slot t_next (all of them already, in ideal mode)."""


class GatedSLAState(SLAState):
    """SLAState whose learned quantities use only outcomes that exist at decision time (FEEDBACK = 'gated')."""

    def __init__(self, N, cfg):
        super().__init__(N, cfg)
        self.t = 0
        self.ev = []                    # (time ms, kind, rows, actions, values)

    def update(self, a, lat, pred, cnt, dl):
        rows = np.arange(len(a))
        t0 = self.t * SLOT_MS
        miss = lat > dl
        off = a > 0
        done = t0 + lat
        known = t0 + np.minimum(lat, dl)
        r_obs = lat - pred[rows, a]
        self.ev += [(done[i], 'resid', i, a[i], r_obs[i]) for i in rows[off]]
        self.ev += [(known[i], 'risk', i, a[i], float(miss[i])) for i in rows]
        self.ev += [(done[i], 'kpi', i, a[i], (float(miss[i]), lat[i] / dl[i])) for i in rows]
        self.kpi[1] = cnt[1:].mean() / SAFETY_CAP
        self.t += 1
        return miss

    def advance(self, t_next):
        now = t_next * SLOT_MS
        w = self.c['ewma']
        due = sorted((e for e in self.ev if e[0] <= now), key=lambda e: e[0])
        self.ev = [e for e in self.ev if e[0] > now]
        fails, ratios = [], []
        for _, kind, i, a, v in due:
            if kind == 'resid':
                self.resid[i, a] = (1 - w) * self.resid[i, a] + w * v
            elif kind == 'risk':
                self.risk[i, a] = (1 - w) * self.risk[i, a] + w * v
            else:
                fails.append(v[0])
                ratios.append(v[1])
        if fails:
            self.kpi[0] = float(np.mean(fails))
            self.kpi[2] = min(max(np.percentile(ratios, 99) - 1.0, 0.0), 1.0)


def make_state(N, cfg):
    return GatedSLAState(N, cfg) if FEEDBACK == 'gated' else SLAState(N, cfg)


def reward(lat, size, dl, risk_a, kpi, lam, N, c):
    rho = PAPER['rho_tau']
    local = lat / size + c['beta_tau'] * ((lat > rho * dl).astype(float) + (lat > dl))
    g = max(kpi[0] - c['fail_star'], 0.0) + max(kpi[1] - c['util_star'], 0.0) + kpi[2]
    early = c['gamma_early'] * max(0.5 - N / 320.0, 0.0)
    return -local - lam * g - c['beta_risk'] * risk_a + early, g


def train(setting, N, run_seed, iters=400, log_every=0, select_every=25, **over):
    """Train one shared SLA-MA-PPO policy at fleet size N; returns networks, curve and timing.

    From the middle of training, the actor is evaluated every ``select_every``
    iterations on the tuning seeds (2000-2009, disjoint from the evaluation
    seeds) and the checkpoint chosen by the tuning rule (lowest latency with
    a miss rate within 0.5 points of the lowest) is returned, which favours
    the baseline when its learning curve oscillates.
    """
    p = dict(PAPER, **{k: v for k, v in over.items() if k in PAPER})
    c = dict(CHOSEN, **{k: v for k, v in over.items() if k not in PAPER})
    rng = np.random.default_rng(RNG_TRAIN_BASE + run_seed)
    M1 = CFG['M'] + 1
    actor = MLP([N_OBS, p['hidden'], p['hidden'], M1], rng, out_scale=0.01)
    critic = MLP([N_OBS, p['hidden'], p['hidden'], 1], rng, out_scale=1.0)
    opt_a = Adam(actor.params(), p['lr'])
    opt_c = Adam(critic.params(), c['critic_lr'])
    T = CFG['SLOTS']
    lam = c['w_glob']
    ret_run, ret_m2, ret_n = 0.0, 0.0, 0
    disc = np.zeros(N)
    curve = []
    ckpts = []
    t0 = time.time()
    for it in range(iters):
        ad = Adapter(setting, N, TRAIN_SEED_INDEX + (it + run_seed * 7) % N_TRAIN_ENVS, T)
        st = make_state(N, c)
        OBS, MSK, ACT, LOGP, REW, VAL = [], [], [], [], [], []
        lat_sum, miss_sum, off_sum = 0.0, 0.0, 0.0
        for t in range(T):
            pred = ad.predicted(t)
            sb = st.proxy(pred)
            m = st.mask(sb, ad.dl)
            o = st.obs(ad, sb, t)
            logits, _ = actor.forward(o)
            pr = masked_probs(logits, m)
            a = samp(pr, rng)
            risk_a = st.risk[np.arange(N), a].copy()
            size = ad.size(t)
            lat, cnt = ad.step(a, t)
            miss = st.update(a, lat, pred, cnt, ad.dl)
            st.advance(t + 1)
            r, g = reward(lat, size, ad.dl, risk_a, st.kpi, lam, N, c)
            if c['dual']:
                lam = max(lam + c['eta_lambda'] * g, 0.0)
            # running scale of discounted returns (reward scaling)
            disc = disc * p['gamma'] + r
            for x in (disc.mean(),):
                ret_n += 1
                d1 = x - ret_run
                ret_run += d1 / ret_n
                ret_m2 += d1 * (x - ret_run)
            v, _ = critic.forward(o)
            OBS.append(o); MSK.append(m); ACT.append(a); LOGP.append(np.log(pr[np.arange(N), a]))
            REW.append(r); VAL.append(v[:, 0])
            lat_sum += lat.mean(); miss_sum += miss.mean(); off_sum += (a > 0).mean()
        scale = np.sqrt(ret_m2 / max(ret_n - 1, 1)) + 1e-8
        REW = np.array(REW) / scale
        VAL = np.array(VAL)
        o_last = st.obs(ad, st.proxy(ad.predicted(T - 1)), T - 1)
        v_last = critic.forward(o_last)[0][:, 0]
        adv = np.zeros_like(REW)
        last = np.zeros(N)
        for t in reversed(range(T)):
            nextv = VAL[t + 1] if t < T - 1 else v_last
            delta = REW[t] + p['gamma'] * nextv - VAL[t]
            last = delta + p['gamma'] * p['gae_lambda'] * last
            adv[t] = last
        ret = (adv + VAL).ravel()
        adv = adv.ravel()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        X = np.concatenate(OBS); MK = np.concatenate(MSK); A = np.concatenate(ACT); LP = np.concatenate(LOGP)
        B = len(A)
        diag = {}
        for _ in range(p['epochs']):
            perm = rng.permutation(B)
            for s in range(0, B, p['minibatch']):
                idx = perm[s:s + p['minibatch']]
                lg, hs = actor.forward(X[idx])
                _, dl, diag = ppo_actor_loss(lg, MK[idx], A[idx], LP[idx], adv[idx], p['clip'], c['ent_coef'])
                opt_a.step(actor.params(), actor.backward(hs, dl))
                v, hs = critic.forward(X[idx])
                opt_c.step(critic.params(), critic.backward(hs, ((v[:, 0] - ret[idx]) / len(idx))[:, None]))
        row = dict(iter=it, lat=lat_sum / T, miss=miss_sum / T, off=off_sum / T, lam=lam,
                   mean_reward=float((REW * scale).mean()), **diag)
        if select_every and it >= iters // 2 and ((it + 1) % select_every == 0 or it == iters - 1):
            cm, _ = evaluate(actor, setting, N, seeds=TUNE_SEEDS, cfg=c)
            ckpts.append(dict(iter=it, lat=cm['lat'], miss=cm['miss'], state=actor.state()))
            row['ckpt_lat'], row['ckpt_miss'] = cm['lat'], cm['miss']
        curve.append(row)
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"  it {it:4d}  lat {row['lat']:6.2f}  miss {100 * row['miss']:5.2f}%  off {row['off']:.3f}  "
                  f"lam {lam:.3f}  H {row.get('entropy', 0):.3f}  {time.time() - t0:6.0f}s", flush=True)
    k = max(1, iters // 10)
    lastm = np.mean([r['lat'] for r in curve[-k:]])
    before = np.mean([r['lat'] for r in curve[-2 * k:-k]]) if iters >= 2 * k else lastm
    chosen_ckpt = None
    if ckpts:
        lo = min(k['miss'] for k in ckpts)
        chosen_ckpt = min([k for k in ckpts if k['miss'] <= lo + 0.005], key=lambda k: k['lat'])
        actor = MLP.from_state(chosen_ckpt['state'])
    return dict(actor=actor, critic=critic, curve=curve, config=dict(paper=p, chosen=c), setting=setting,
                checkpoints=[{k: v for k, v in x.items() if k != 'state'} for x in ckpts],
                chosen_iter=None if chosen_ckpt is None else chosen_ckpt['iter'],
                N_train=N, run_seed=run_seed, carry=LOCAL_CARRY or 'none', feedback=FEEDBACK,
                plateau=bool(abs(lastm - before) <= 0.01 * abs(before)),
                plateau_rel_change=float(abs(lastm - before) / max(abs(before), 1e-12)),
                seconds=time.time() - t0)


def run_eval(actor, setting, N, seed, cfg=None):
    """One evaluation episode on archived seed index ``seed`` (metrics as gipa_core.run / hetero.run_het)."""
    c = dict(CHOSEN, **(cfg or {}))
    T = CFG['SLOTS']
    ad = Adapter(setting, N, seed, T)
    st = make_state(N, c)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    out = dict(mS=[], mB=[], miss=[], lat=[], p95=[], off=[], offS=[], offB=[], maxl=[], fallback=[])
    for t in range(T):
        pred = ad.predicted(t)
        sb = st.proxy(pred)
        m = st.mask(sb, ad.dl)
        o = st.obs(ad, sb, t)
        logits, _ = actor.forward(o)
        a = samp(masked_probs(logits, m), rg)
        screened = ~(np.isfinite(sb) & (sb <= PAPER['rho_tau'] * ad.dl[:, None]) & (st.risk <= PAPER['eta'])).any(1)
        lat, cnt = ad.step(a, t)
        miss = st.update(a, lat, pred, cnt, ad.dl)
        st.advance(t + 1)
        if t >= CFG['WARM']:
            S = ad.isS
            out['mS'].append(miss[S].mean() if S.any() else 0.0)
            out['mB'].append(miss[~S].mean() if (~S).any() else 0.0)
            out['miss'].append(miss.mean())
            out['lat'].append(lat.mean())
            out['p95'].append(np.percentile(lat, 95))
            out['off'].append((a > 0).mean())
            out['offS'].append((a[S] > 0).mean() if S.any() else 0.0)
            out['offB'].append((a[~S] > 0).mean() if (~S).any() else 0.0)
            out['maxl'].append(cnt[1:].max())
            out['fallback'].append(screened.mean())
    return {k: safe_mean(v) for k, v in out.items()}


def evaluate(actor, setting, N, seeds=None, cfg=None):
    seeds = range(CFG['NSEED']) if seeds is None else seeds
    rs = [run_eval(actor, setting, N, s, cfg) for s in seeds]
    return {k: safe_mean([r[k] for r in rs]) for k in rs[0]}, rs
