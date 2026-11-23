"""
qlearning.py
============
Configurable tabular Q-learning baseline for workplan experiment E4,
answering Reviewer 2 comment 4 ("the Q-learning baseline appears to rely on
a single training seed ... mention training details and evaluate it across
multiple independent training runs").

STATUS: written for the first revision of manuscript ITS-2026-08-0579. Not
part of the submitted v1.0.0 results.

Relation to the archived learner
--------------------------------
``gipa_core.QL`` is left untouched so v1.0.0 results still reproduce. The
class :class:`QLearner` here generalises it. With
``QLearner(train_seed=5999, env_offset=0, gamma=0.0, model='affine')`` and
``reset=False`` at evaluation it reproduces ``gipa_core.QL`` exactly
(verified in :func:`check_archived`). The revision fixes, each controlled by
an argument:

* **Disjoint training environments** (``env_offset``): the archived learner
  trains on seed indices ``step % 7`` = 0..6, which are also evaluation
  seeds 0..6. The default offset 3000 trains on seed indices 3000..3006
  (environment RNG 4000..4006), disjoint from evaluation (0..29), from
  MAPPO training (1000..1049) and from MAPPO tuning (2000..2009).
* **Episode reset** (``reset=True`` in :func:`run_ql`): the previous-slot
  load is cleared at the start of every evaluation episode. In the archive
  it carries over from the previous episode.
* **Multiple training runs** (``train_seed``): exploration RNG per run;
  E4 uses 6000..6009.
* **Discount** (``gamma``): the archived update has no next-state term
  (gamma = 0, a contextual bandit with running-average values). ``gamma``
  > 0 adds the standard bootstrapped target r + gamma * max_a' Q(s', a'),
  where s' is the vehicle's class with the load bucket of the next slot.
* **Latency model** (``model``): 'affine' (submitted) or 'ps' (Eq. 5, E2).

Specification (unchanged from the archive)
------------------------------------------
One Q-table of shape (2 classes, 4 load buckets, M + 1 actions), shared by
all vehicles. Bucket = min(int(4 * largest RSU share of the fleet in the
previous slot), 3). Reward = -(latency / deadline) - 3 * miss. Learning rate
0.1. Exploration epsilon = max(0.05, 1 - step / (0.7 * steps)). One training
step = one slot in which all N vehicles act; their N updates are applied in
sequence. 15,000 steps. At evaluation: greedy action plus 2% uniformly
random actions.

Diagnostics recorded during training
------------------------------------
Every ``log_every`` steps: mean reward of that slot, mean latency, offload
ratio, and the largest absolute change in any Q-value since the previous
log point (convergence check).
"""

import numpy as np

from gipa_core import CFG, Env, safe_mean, RNG_ACT_BASE

ENV_OFFSET_DEFAULT = 3000


def make_env(N, seed, R, model):
    """'affine' -> gipa_core.Env; 'ps' -> calibration.PSEnv (Eq. 5)."""
    if model == 'affine':
        return Env(N, seed, R)
    from calibration import PSEnv, make_calibration
    return PSEnv(N, seed, R, make_calibration())


class QLearner:
    """Tabular learner over (class, load bucket); see the module docstring."""

    def __init__(self, n, train_seed=5999, env_offset=ENV_OFFSET_DEFAULT, gamma=0.0,
                 model='affine', n_envs=7):
        self.Q = np.zeros((2, 4, n))
        self.n = n
        self.mu = None
        self.train_seed = train_seed
        self.env_offset = env_offset
        self.gamma = gamma
        self.model = model
        self.n_envs = n_envs
        self.curve = []

    def bucket(self):
        """Load bucket 0-3 from the previous slot's largest RSU share."""
        return 0 if self.mu is None else min(int(self.mu[1:].max() * 4), 3)

    def train(self, N, R, steps=15000, pen=3.0, lr=0.10, log_every=500):
        """Train at fleet size N in regime R. Mirrors gipa_core.QL.train."""
        rg = np.random.default_rng(self.train_seed)
        self.mu = np.full(self.n, 1. / self.n)
        Q_prev = self.Q.copy()
        for st in range(steps):
            eps = max(0.05, 1 - st / (0.7 * steps))
            env = make_env(N, self.env_offset + st % self.n_envs, R, self.model)
            b = self.bucket()
            cls = (~env.isS[:N]).astype(int)
            greedy_a = self.Q[cls, b].argmax(1)
            rnd = rg.integers(0, self.n, N)
            a = np.where(rg.random(N) < eps, rnd, greedy_a)
            lat, c = env.lat(a, N)
            miss = lat > env.dl[:N]
            rew = -(lat / env.dl[:N]) - pen * miss
            if self.gamma > 0:
                b_next = min(int((c / N)[1:].max() * 4), 3)
                for i in range(N):
                    target = rew[i] + self.gamma * self.Q[cls[i], b_next].max()
                    self.Q[cls[i], b, a[i]] += lr * (target - self.Q[cls[i], b, a[i]])
            else:
                for i in range(N):
                    self.Q[cls[i], b, a[i]] += lr * (rew[i] - self.Q[cls[i], b, a[i]])
            self.mu = c / N
            if log_every and (st % log_every == log_every - 1 or st == steps - 1):
                self.curve.append(dict(step=st + 1, reward=float(rew.mean()), lat=float(lat.mean()),
                                       off=float((a > 0).mean()), eps=float(eps),
                                       max_dQ=float(np.abs(self.Q - Q_prev).max())))
                Q_prev = self.Q.copy()

    def act(self, env, q, cb, rg, N):
        """Greedy action from the table with 2% random exploration."""
        b = self.bucket()
        cls = (~env.isS[:N]).astype(int)
        a = self.Q[cls, b].argmax(1)
        f = rg.random(N) < 0.02
        a[f] = rg.integers(0, self.n, f.sum())
        self.mu = np.bincount(a, minlength=self.n) / N
        return a

    def greedy_policy(self):
        """Greedy action per (class, bucket): 0 = local, j = RSU j."""
        return self.Q.argmax(2).tolist()


def run_ql(ql, N, seed, R, reset=True, model='affine'):
    """One evaluation episode on archived seed index ``seed``.

    Same environment seed, action seed, slots, warm-up and metrics as
    ``gipa_core.run``. ``reset=True`` clears the previous-slot load at the
    start of the episode (the revision fix); ``reset=False`` keeps the
    archived carry-over behaviour.
    """
    if reset:
        ql.mu = None
    env = make_env(N, seed, R, model)
    rg = np.random.default_rng(RNG_ACT_BASE + seed)
    out = dict(mS=[], mB=[], lat=[], fair=[], off=[], offS=[], offB=[], en=[], p95=[], maxl=[])
    for t in range(CFG['SLOTS']):
        q = env.q(N)
        a = ql.act(env, q, None, rg, N)
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
    return {k: safe_mean(v) for k, v in out.items()}


def evaluate(ql, N, R, reset=True, model='affine'):
    """Evaluate on the 30 archived seeds; returns (means, per-seed list)."""
    rs = [run_ql(ql, N, s, R, reset, model) for s in range(CFG['NSEED'])]
    return {k: safe_mean([r[k] for r in rs]) for k in rs[0]}, rs


def two_level_bootstrap(M, reps=10000, seed=20260804, level=0.95, chunk=1000):
    """Percentile CI for the grand mean of a (training runs x evaluation seeds) matrix.

    Each replicate resamples training runs with replacement, then, for each
    selected run, resamples evaluation seeds with replacement. This captures
    both training-run and evaluation-seed variability. NaN entries are
    ignored. Vectorised in chunks of ``chunk`` replicates. Returns
    (mean, lo, hi).
    """
    M = np.asarray(M, dtype=float)
    rg = np.random.default_rng(seed)
    T, E = M.shape
    stats = []
    done = 0
    while done < reps:
        k = min(chunk, reps - done)
        rows = rg.integers(0, T, (k, T))
        cols = rg.integers(0, E, (k, T, E))
        stats.append(np.nanmean(M[rows[:, :, None], cols], axis=(1, 2)))
        done += k
    stats = np.concatenate(stats)
    a = (1 - level) / 2
    return float(np.nanmean(M)), float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def check_archived(N=160, R=None):
    """True if QLearner in the archived configuration reproduces gipa_core.QL exactly.

    Trains both with the archived settings (seed 5999, environments 0..6,
    gamma 0), compares the Q-tables, then evaluates both with the archived
    carry-over behaviour on all 30 seeds and compares every metric.
    """
    from gipa_core import QL, REG, agg
    R = R or REG['A']
    old = QL(CFG['M'] + 1)
    old.train(N, R)
    new = QLearner(CFG['M'] + 1, train_seed=5999, env_offset=0, gamma=0.0, model='affine')
    new.train(N, R, log_every=0)
    same_q = np.array_equal(old.Q, new.Q)
    m_old, _ = agg('ql', N, R, ql=old)          # starts from the post-training load state
    m_new, _ = evaluate(new, N, R, reset=False)  # same starting state, same carry-over
    same_m = all((m_old[k] == m_new[k]) or (np.isnan(m_old[k]) and np.isnan(m_new[k])) for k in m_old)
    return bool(same_q and same_m)
