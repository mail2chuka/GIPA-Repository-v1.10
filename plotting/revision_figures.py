"""
revision_figures.py
===================
Summary figures for the first revision of manuscript ITS-2026-08-0579, one
per reviewer concern, drawn from the revision result files. It never runs a
simulator.

STATUS: new for the revision. The v1.0.0 figures (Figs. 3-8) are still drawn
by plotting/generate_figures.py.

WHERE TO EDIT THE WORDING
-------------------------
Every word that appears in a figure (legend entries, panel titles, axis
labels, tick labels, notes) is set in the block marked "EDIT HERE" below:

  LEGEND        method names, used in every legend and bar label.
  TEXT['Rn']    titles, axis labels and notes of figure Rn.
  PANEL_TITLES  'full'   -> "(a) Affine latency model"
                'letter' -> "(a)" only (the description stays in the caption)
                'none'   -> no panel titles
                (also settable per run: --titles letter)

Legend position and number of columns are set where each legend is drawn;
those lines are marked "# EDIT: legend position". After editing, rerun this
script and then the manuscript build so Word picks up the new images.

Note: the plotted values in results/res_revision_figures.json are keyed by
the legend labels, so editing a label also renames its key in that file.

Figures (saved to figures/revision/ as PNG at 300 dpi, PDF, SVG and TIFF)
------------------------------------------------------------------------
Manuscript figure number in brackets (first revision).

R1  fig_r1_main_comparison     [3]  E2/E6/E7 (R2-2, R1-3, R1-4): latency vs N,
                                    affine and Eq. (5) models
R2  fig_r2_prediction_error    [4]  E1 (R2-1): misses by error arm and guard
                                    band (N = 80); prior fleet-size
                                    misspecification (N = 160)
R3  fig_r3_capacity            [5]  E3 (R2-3): selected RSU load against
                                    capacity; rejections under admission
                                    control (N = 160)
R4  fig_r4_qlearning_runs      [8]  E4 (R2-4): latency of each of 10
                                    Q-learning training runs (N = 160)
R5  fig_r5_asynchronous        [6]  E5 (R1-6): latency and misses with
                                    asynchronous arrivals, persistent queues
R6  fig_r6_learning_automata   [9]  E6 (R1-3): LA learning curve; density
                                    change N = 80 -> 240 -> 80
R7  fig_r7_asymmetric_rsus     [7]  E9 (core claim): herding ratio at the fast
                                    RSU; latency vs N
R8  fig_r8_decision_cost      [10]  E8 (R1-8): decision time, runtime payload
R9  fig_r9_heterogeneous_tasks[11]  E12: heterogeneous tasks; amplification on
                                    the plan versus per-task ranking

TIFF files are 600 dpi, RGB, LZW-compressed. plotting/export_submission_tiffs.py
copies them, with Figs. 1 and 2, to figures/submission/Fig1.tif ... Fig11.tif.

Statistics
----------
Error bars are seed-level 95% percentile-bootstrap intervals from
src/stats.py (10,000 resamples). Each figure uses its own generator seeded
with 20260804, drawn in the order the series are plotted, so a figure can be
regenerated alone. MAPPO intervals use the per-seed mean over its 5
training runs; Q-learning shows every training run instead of an interval.
Overall miss rates are 0.3 x safety + 0.7 x background (the class shares).
Every plotted value and interval is also written to
results/res_revision_figures.json, so numbers in the text can be checked
against the figures.

Colour and marks
----------------
One fixed colour and marker per method in every figure (validated
categorical palette, adjacent CVD separation >= 9.1 OKLab x 100): GIPA with
planned prior blue (circle), GIPA with heuristic prior orange (square),
learning automata aqua (triangle), MAPPO yellow (diamond), RBR magenta
(inverted triangle), GIPA without diffusion / masked softmax green
(pentagon), Q-learning violet (hexagon), greedy / near-live reference red
(cross); all-local is a grey dashed reference line. Every multi-series panel
has a legend; markers differ so identity never depends on colour alone.

Run (from the repository root)
------------------------------
    PYTHONPATH=src python plotting/revision_figures.py                 # all
    PYTHONPATH=src python plotting/revision_figures.py --figure 5      # one
    PYTHONPATH=src python plotting/revision_figures.py --titles letter # "(a)" only
"""

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from stats import Bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OUT = ROOT / 'figures' / 'revision'
PI_S = 0.3
DENS = [80, 160, 240, 320]

# ======================================================================
# EDIT HERE: wording of every figure
# ======================================================================

# Panel titles: 'full', 'letter' or 'none' (see the module docstring).
PANEL_TITLES = 'full'

# Method names. One entry per method; used in every legend and bar label.
LEGEND = {
    'plan':     'GIPA, planned prior',
    'gipa':     'GIPA, heuristic prior',
    'la':       'Learning automata',
    'mappo':    'MAPPO',
    'rbr':      'Reactive best response',
    'nodiff':   'GIPA, heuristic prior, no diffusion',
    'msoft':    'Masked softmax',
    'ql':       'Q-learning',
    'greedy':   'Greedy',
    'local':    'All-local',
    # methods that appear in one figure only
    'msoft_prior': 'Prior-adjusted masked softmax',
    'indep':    'Independent softmax',
    'price':    'Congestion pricing',
    'entreg':   'Entropy-regularised',
}

# Labels shared by several figures.
X_FLEET = 'Fleet size N (vehicles)'
Y_LATENCY = 'Mean latency (ms)'
Y_MISSES = 'Misses, all tasks (%)'

# Per-figure wording. Panel titles are (letter, description) pairs.
TEXT = {
    'R1': dict(                                    # manuscript Fig. 3
        title_a=('(a)', 'Affine latency model'),
        title_b=('(b)', 'Eq. (5), class-dependent compute'),
        la='Learning automata (500-slot warm-up)',
        xlabel=X_FLEET, ylabel=Y_LATENCY),
    'R2': dict(                                    # manuscript Fig. 4
        title_a=('(a)', 'Prediction error arms, N = 80'),
        title_b=('(b)', 'Fleet-size error in the prior, N = 160'),
        # x tick labels of panel (a), in plotting order
        arms=['No\nerror', 'Score\nσ=0.3', 'Mask\nσ=0.3', 'Both\nσ=0.3', 'Both,\n10%\nguard', 'Both,\n20%\nguard'],
        ylabel_a='Deadline misses, all tasks (%)',
        note_a='GIPA, planned prior:\nno misses in any\ncondition',   # set to '' to drop
        xlabel_b='Anticipated / true fleet size', ylabel_b=Y_LATENCY,
        ref_b='All-local'),
    'R3': dict(                                    # manuscript Fig. 5
        title_a=('(a)', 'Largest selected RSU load, no admission'),
        title_b=('(b)', 'Rejected under admission control'),
        xlabel_a='Tasks per RSU per slot',
        xlabel_b='Rejected offloads (%)',
        cap_fmt='{name} = {value}'),               # capacity line labels, e.g. "Q_S = 31"
    'R4': dict(                                    # manuscript Fig. 8
        title=('', 'Q-learning across 10 training runs, Regime A'),
        # x tick groups; "{n}" is replaced by the number of runs that ended all-local
        groups=['γ = 0\naffine', 'γ = 0.9\naffine', 'γ = 0\nEq. (5)'],
        group_count='{n}/10 all-local',
        run='One training run', mean='Mean of 10 runs',
        archived='Archived single run (seed 5999)',
        local='All-local (18.38 ms)',
        plan='GIPA, planned prior, affine (16.69 ms)',
        ylabel='Mean latency at N = 160 (ms)'),
    'R5': dict(                                    # manuscript Fig. 6
        title_a=('(a)', 'Mean latency (runs capped at 60 ms)'),
        title_b=('(b)', 'Deadline misses'),
        ylabel_a=Y_LATENCY, ylabel_b=Y_MISSES,
        xlabel='Fleet size N (one task per vehicle per 20 ms)',
        ticks={55: '55\nρ=0.5', 80: '80', 87: '87\nρ=0.8', 104: '104\nρ=0.95', 160: '160', 240: '240'},
        jsq='Near-live shortest delay (1 ms backlog)',
        rbr20='Reactive best response, 20 ms broadcast',
        rbr100='Reactive best response, 100 ms broadcast',
        # footer note; set to '' to drop it. {note} is filled from the data.
        footer=('Unbounded backlog growth: reactive best response with a 100 ms broadcast in {note};\n'
                'GIPA (either prior) and learning automata in no run.')),
    'R5b': dict(                                   # manuscript Fig. 9 (second revision round)
        title_a=('(a)', 'Mean latency (runs capped at 60 ms)'),
        title_b=('(b)', 'Deadline misses'),
        ylabel_a=Y_LATENCY, ylabel_b=Y_MISSES,
        xlabel='Fleet size N (one task per vehicle per 20 ms)',
        ticks={55: '55\nρ=0.5', 80: '80', 87: '87\nρ=0.8', 104: '104\nρ=0.95', 160: '160', 240: '240'},
        plan_own='GIPA, planned prior, own queue in mask',
        plan_e5='GIPA, planned prior ignoring local queues',
        gipa_own='GIPA, heuristic prior, own queue in mask',
        jsq='Near-live shortest delay (1 ms backlog)',
        rbr20='Reactive best response, 20 ms broadcast',
        # footer note; set to '' to drop it. {note} is filled from the data.
        footer=''),  # e.g. 'All-local execution: unbounded local backlog in every run.{note}'
    'R6': dict(                                    # manuscript Fig. 9
        title_a=('(a)', 'Learning curve, N = 160'),
        title_b=('(b)', 'Fleet-size change, Regime A'),
        la='Learning automata (L_R-εP)',
        ref_gipa='GIPA, heuristic prior = all-local (18.38)',
        ref_plan='GIPA, planned prior (16.69)',
        warmup='end of archived\n25-slot warm-up',
        xlabel_a='Slot (25-slot block end)', ylabel_a=Y_LATENCY,
        xlabel_b='Slot', ylabel_b='Mean latency (ms, 10-slot average)',
        phases=['N = 80', 'N = 240', 'N = 80'],
        plan_fresh='GIPA, planned prior, re-planned for N',
        gipa_fresh='GIPA, heuristic prior refreshed for N',
        gipa_stale='GIPA, heuristic prior for N = 80',
        la_lrep='Learning automata (L_R-εP)'),
    'R7': dict(                                    # manuscript Fig. 7
        title_a=('(a)', 'Herding onto the fast RSU, N = 80'),
        title_b=('(b)', 'Latency with one RSU 30% faster'),
        xlabel_a='Fast-RSU load / capacity-weighted share',
        fair='fair share\nfor capacity', all_fast='every offload\non RSU 1',
        gipa_cap='GIPA, capacity-recalibrated heuristic prior',
        xlabel_b=X_FLEET, ylabel_b=Y_LATENCY),
    'R8': dict(                                    # manuscript Fig. 10
        title_a=('(a)', 'Decision time'),
        title_b=('(b)', 'Communication'),
        mappo='MAPPO actor',
        per_vehicle='({v:.0f} μs per vehicle)',
        notes={'rbr': 'per slot, after a load broadcast', 'greedy': 'centralised'},
        slot='20 ms slot',
        xlabel_a='Fleet decision time per slot, N = 160 (μs)',
        xlabel_b='Runtime application payload (B/s, log scale)',
        none='0 (none)',
        # footer: machine details are filled in from the result file; set to '' to drop it
        footer='Timings: {machine};\none thread; {tail}.'),
    'R9': dict(                                    # manuscript Fig. 11
        title_a=('(a)', 'Mean latency'),
        title_b=('(b)', 'Deadline misses'),
        plan='GIPA, planned prior, no amplification',
        plan_rank='GIPA, planned prior + per-task ranking',
        gipa_plan='GIPA, planned prior, amplification',
        ylabel_a=Y_LATENCY, ylabel_b=Y_MISSES, xlabel=X_FLEET),
}

# Output formats and raster resolution.
FORMATS = ['png', 'pdf', 'svg', 'tif']
PNG_DPI = 300
TIFF_DPI = 600

# ======================================================================
# end of EDIT HERE
# ======================================================================

INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
STYLE = {
    'plan':   dict(color='#2a78d6', marker='o'),
    'gipa':   dict(color='#eb6834', marker='s'),
    'la':     dict(color='#1baf7a', marker='^'),
    'mappo':  dict(color='#eda100', marker='D'),
    'rbr':    dict(color='#e87ba4', marker='v'),
    'nodiff': dict(color='#008300', marker='p'),
    'msoft':  dict(color='#008300', marker='p'),
    'ql':     dict(color='#6250d6', marker='h'),
    'greedy': dict(color='#e34948', marker='X'),
    'local':  dict(color=INK2, marker=None),
}
for _k, _st in STYLE.items():
    _st['label'] = LEGEND[_k]

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'lines.linewidth': 1.5, 'lines.markersize': 5, 'legend.frameon': False,
    'savefig.dpi': PNG_DPI, 'font.family': 'DejaVu Sans',
})

VALUES = {}


def load(name):
    return json.load(open(RES / name))


def ci(bs, vals):
    m, (lo, hi), n = bs.ci(vals)
    return m, lo, hi


def title(ax, t):
    """Set a panel title from a (letter, description) pair according to PANEL_TITLES."""
    letter, text = t
    if PANEL_TITLES == 'none':
        return
    s = letter if PANEL_TITLES == 'letter' else ' '.join(x for x in (letter, text) if x)
    if s:
        ax.set_title(s, loc='left')


def line(ax, bs, key, xs, per_x, label=None, record=None, **kw):
    """Plot mean +/- CI for a series; per_x is a list of per-seed value lists (None to skip)."""
    st = dict(STYLE[key])
    st.update(kw)
    pts = []
    for x, vals in zip(xs, per_x):
        if vals is None:
            continue
        m, lo, hi = ci(bs, vals)
        pts.append((x, m, lo, hi))
    if not pts:
        return
    X, M, L, H = map(np.array, zip(*pts))
    open_marker = st.get('mfc') == 'white'
    ax.errorbar(X, M, yerr=[M - L, H - M], color=st['color'], marker=st['marker'], ms=5,
                lw=1.5, ls=st.get('ls', '-'), capsize=2, elinewidth=0.8, label=label or st['label'],
                mfc='white' if open_marker else st['color'], mec=st['color'] if open_marker else 'white',
                mew=1.0 if open_marker else 0.6, zorder=3)
    if record is not None:
        record[label or st['label']] = [dict(x=float(x), mean=float(m), ci95=[float(lo), float(hi)]) for x, m, lo, hi in pts]


def ref_line(ax, y, text, color=INK2, below=False):
    ax.axhline(y, color=color, ls='--', lw=1, zorder=1)
    ax.annotate(text, (1.0, y), xycoords=('axes fraction', 'data'), xytext=(-2, -3 if below else 3),
                textcoords='offset points', ha='right', va='top' if below else 'bottom', fontsize=7, color=color)


def save_tiff(fig, path, dpi=TIFF_DPI):
    """Write an RGB, LZW-compressed TIFF at `dpi` (no alpha channel, white background)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    Image.open(buf).convert('RGB').save(path, format='TIFF', compression='tiff_lzw', dpi=(dpi, dpi))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        if ext == 'tif':
            save_tiff(fig, OUT / f'{stem}.tif')
        else:
            fig.savefig(OUT / f'{stem}.{ext}', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', OUT / stem)


def overall(recs):
    return [PI_S * r['mS'] + (1 - PI_S) * r['mB'] for r in recs]


# ----------------------------------------------------------------------
def fig_r1():
    """Main comparison under both latency models (E2, E6, E7). Manuscript Fig. 3."""
    T = TEXT['R1']
    seed = load('res_main_seed_level.json')
    plan = load('res_planner.json')['per_seed']
    cal = load('res_calibration.json')['per_seed']
    la = load('res_la.json')
    mp = load('res_mappo.json')['groups']
    rec = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, model, ttl in [(axes[0], 'affine', T['title_a']), (axes[1], 'ps', T['title_b'])]:
        bs = Bootstrap()
        r = rec.setdefault(model, {})

        def mappo(N):
            grp = mp[f'{model}|base|central|A' + ('|deadline' if N >= 240 else '')]
            runs = [g for g in grp if g['N_train'] == N]
            if not runs:
                return None
            return list(np.mean([[s['lat'] for s in g['eval']['per_seed']] for g in runs], 0))
        if model == 'affine':
            local = [[x['lat'] for x in seed['A'][str(N)]['local']] for N in DENS]
            line(ax, bs, 'plan', DENS, [[x['lat'] for x in plan['A'][str(N)]['plan_z2']] for N in DENS], record=r)
            line(ax, bs, 'gipa', DENS, [[x['lat'] for x in seed['A'][str(N)]['gipa']] for N in DENS], record=r)
        else:
            local = [[x['lat'] for x in cal['A'][str(N)]['local']] for N in DENS]
            line(ax, bs, 'plan', DENS, [[x['lat'] for x in cal['A'][str(N)]['plan_z2|class']] for N in DENS], record=r)
            line(ax, bs, 'gipa', DENS, [[x['lat'] for x in cal['A'][str(N)]['gipa|class']] for N in DENS], record=r)
        line(ax, bs, 'la', DENS, [[x['lat'] for x in la['per_seed'][f'A|{model}|500|{N}|best']] for N in DENS],
             label=T['la'], record=r)
        line(ax, bs, 'mappo', DENS, [mappo(N) for N in DENS], record=r)
        # The planned prior and MAPPO coincide; redraw the former's markers open and on top so both stay visible.
        pts = r[STYLE['plan']['label']]
        ax.plot([p['x'] for p in pts], [p['mean'] for p in pts], ls='none', marker='o', ms=8, mfc='none',
                mec=STYLE['plan']['color'], mew=1.4, zorder=5)
        lm = [np.mean(v) for v in local]
        ax.plot(DENS, lm, color=INK2, ls='--', lw=1, label=LEGEND['local'], zorder=1)
        r[LEGEND['local']] = [dict(x=N, mean=float(m)) for N, m in zip(DENS, lm)]
        title(ax, ttl)
        ax.set_xlabel(T['xlabel'])
        ax.set_xticks(DENS)
    axes[0].set_ylabel(T['ylabel'])
    axes[0].set_ylim(14.5, 20.2)
    h, lab = axes[0].get_legend_handles_labels()
    order = [lab.index(x) for x in lab if x != LEGEND['local']] + [lab.index(LEGEND['local'])]
    # EDIT: legend position (one shared legend below both panels, 5 columns)
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    VALUES['R1'] = rec
    save(fig, 'fig_r1_main_comparison')


def fig_r2():
    """Prediction error (E1). Manuscript Fig. 4."""
    T = TEXT['R2']
    p = load('res_prediction.json')['conditions']
    by = {(c['N'], c['name']): c for c in p}
    rec = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    bs = Bootstrap()
    conds = list(zip(['baseline', 'score_sigma0.30', 'mask_sigma0.30', 'both_sigma0.30',
                      'both_sigma0.30_guard0.10', 'both_sigma0.30_guard0.20'], T['arms']))
    pols = [('msoft', 'msoft'), ('gipa', 'gipa'), ('plan', 'plan')]
    w = 0.26
    x0 = np.arange(len(conds))
    for i, (pol, key) in enumerate(pols):
        st = STYLE[key]
        ms, los, his = [], [], []
        for name, _ in conds:
            v = [100 * x for x in overall(by[(80, name)]['per_seed'][pol])]
            m, lo, hi = ci(bs, v)
            ms.append(m)
            los.append(lo)
            his.append(hi)
        ms, los, his = map(np.array, (ms, los, his))
        ax.bar(x0 + (i - 1) * w, ms, w * 0.92, color=st['color'], label=st['label'], zorder=2)
        ax.errorbar(x0 + (i - 1) * w, ms, yerr=[ms - los, his - ms], fmt='none', ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
        rec.setdefault('a_miss_pct_N80', {})[st['label']] = [dict(condition=c[0], mean=float(m), ci95=[float(lo), float(hi)])
                                                             for c, m, lo, hi in zip(conds, ms, los, his)]
    ax.set_xticks(x0)
    ax.set_xticklabels([c[1] for c in conds])
    ax.set_ylabel(T['ylabel_a'])
    title(ax, T['title_a'])
    ax.legend(loc='upper right')  # EDIT: legend position, panel (a)
    if T['note_a']:
        ax.annotate(T['note_a'], (0.03, 0.55), xycoords='axes fraction',
                    fontsize=6.5, color=STYLE['plan']['color'], va='top')
    ax = axes[1]
    bs = Bootstrap()
    fs = [0.5, 0.75, 1.0, 1.25, 1.5]
    names = {0.5: 'prior_f0.50', 0.75: 'prior_f0.75', 1.0: 'baseline', 1.25: 'prior_f1.25', 1.5: 'prior_f1.50'}
    r = rec.setdefault('b_prior_misspec_N160', {})
    for pol, key in [('plan', 'plan'), ('gipa', 'gipa'), ('msoft_prior', 'msoft')]:
        lab = LEGEND['msoft_prior'] if pol == 'msoft_prior' else None
        line(ax, bs, key, fs, [[x['lat'] for x in by[(160, names[f])]['per_seed'][pol]] for f in fs], label=lab, record=r)
    ref_line(ax, 18.38, T['ref_b'])
    ax.set_xticks(fs)
    ax.set_xticklabels(['0.5', '0.75', '1', '1.25', '1.5'])
    ax.set_xlabel(T['xlabel_b'])
    ax.set_ylabel(T['ylabel_b'])
    title(ax, T['title_b'])
    ax.legend(loc='upper right')  # EDIT: legend position, panel (b)
    fig.tight_layout()
    VALUES['R2'] = rec
    save(fig, 'fig_r2_prediction_error')


def fig_r3():
    """Capacity compliance (E3). Manuscript Fig. 5."""
    T = TEXT['R3']
    cap = load('res_capacity.json')
    Q = cap['capacities']['A']
    ps = cap['per_seed']['A']['160']
    pols = [(k, LEGEND[k]) for k in ['greedy', 'indep', 'rbr', 'price', 'msoft', 'entreg', 'gipa', 'plan']]
    rec = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    y = np.arange(len(pols))[::-1]
    for ax, metric, mode, ttl, xl in [(axes[0], 'maxl_sel', 'soft', T['title_a'], T['xlabel_a']),
                                      (axes[1], 'rej', 'ac', T['title_b'], T['xlabel_b'])]:
        bs = Bootstrap()
        scale = 100 if metric == 'rej' else 1
        ms, los, his = [], [], []
        for pol, _ in pols:
            v = [scale * (s[metric] or 0.0) for s in ps[pol][mode]]
            m, lo, hi = ci(bs, v)
            ms.append(m)
            los.append(lo)
            his.append(hi)
        ms, los, his = map(np.array, (ms, los, his))
        colors = [STYLE['plan']['color'] if p == 'plan' else STYLE['gipa']['color'] if p == 'gipa' else '#9b9a95' for p, _ in pols]
        ax.barh(y, ms, 0.62, color=colors, zorder=2)
        ax.errorbar(ms, y, xerr=[ms - los, his - ms], fmt='none', ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
        for yy, m in zip(y, ms):
            ax.annotate(f'{m:.0f}', (m, yy), xytext=(3, 0), textcoords='offset points', va='center', fontsize=7, color=INK)
        title(ax, ttl)
        ax.set_xlabel(xl)
        rec[metric] = {lab: dict(mean=float(m), ci95=[float(lo), float(hi)]) for (_, lab), m, lo, hi in zip(pols, ms, los, his)}
        if metric == 'maxl_sel':
            for qv, nm in [(Q['Q_S'], 'Q_S'), (Q['Q_B'], 'Q_B')]:
                ax.axvline(qv, color=INK2, ls='--', lw=1)
                ax.annotate(T['cap_fmt'].format(name=nm, value=qv), (qv, y[-1] - 0.45), xytext=(3, 0),
                            textcoords='offset points', fontsize=7, color=INK2)
            ax.set_xlim(0, 185)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _, lab in pols])  # method names act as the legend
    fig.tight_layout()
    VALUES['R3'] = rec
    save(fig, 'fig_r3_capacity')


def fig_r4():
    """Q-learning over 10 training runs (E4). Manuscript Fig. 8."""
    T = TEXT['R4']
    q = load('res_qlearning.json')
    groups = list(zip([('affine', 0.0), ('affine', 0.9), ('ps', 0.0)], T['groups']))
    rec = {}
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    rg = np.random.default_rng(1)
    ticks = []
    for i, ((model, gamma), lab) in enumerate(groups):
        runs = [r for r in q['runs'] if r['regime'] == 'A' and r['model'] == model and r['gamma'] == gamma
                and r['N_train'] == 160]
        lat = np.array([r['eval']['160']['mean']['lat'] for r in runs])
        ax.scatter(i + rg.uniform(-0.12, 0.12, len(lat)), lat, s=22, color=STYLE['ql']['color'],
                   marker=STYLE['ql']['marker'], edgecolor='white', linewidth=0.5, zorder=3,
                   label=T['run'] if i == 0 else None)
        ax.plot([i - 0.25, i + 0.25], [lat.mean()] * 2, color=INK, lw=1.5, zorder=4, label=T['mean'] if i == 0 else None)
        n_local = int(np.sum(np.abs(lat - 18.2) < 0.1))
        ticks.append(f'{lab}\n' + T['group_count'].format(n=n_local))
        rec[lab.replace('\n', ' ')] = dict(runs=lat.tolist(), mean=float(lat.mean()))
    ax.scatter([0], [18.2003], marker='*', s=70, color=INK, zorder=5, label=T['archived'])
    ax.axhline(18.38, color=INK2, ls='--', lw=1, zorder=1, label=T['local'])
    ax.axhline(16.69, color=STYLE['plan']['color'], ls='--', lw=1, zorder=1, label=T['plan'])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(ticks)
    ax.set_ylabel(T['ylabel'])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(15.5, 39)
    ax.legend(loc='upper center', fontsize=6.3)  # EDIT: legend position
    title(ax, T['title'])
    fig.tight_layout()
    VALUES['R4'] = rec
    save(fig, 'fig_r4_qlearning_runs')


def fig_r5():
    """Asynchronous arrivals with persistent queues (E5). Manuscript Fig. 6."""
    T = TEXT['R5']
    a = load('res_async.json')
    ps = a['per_seed']
    Ns = a['fleets']
    xs = list(range(len(Ns)))
    series = [('plan', 'plan', None), ('la', 'la', None), ('gipa_slot', 'gipa', None),
              ('jsq_live', 'greedy', T['jsq']),
              ('rbr_b20', 'rbr', T['rbr20']),
              ('rbr_b100', 'rbr', T['rbr100'])]
    rec = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, metric, ttl, yl in [(axes[0], 'lat', T['title_a'], T['ylabel_a']),
                                (axes[1], 'miss', T['title_b'], T['ylabel_b'])]:
        bs = Bootstrap()
        r = rec.setdefault(metric, {})
        for pol, key, lab in series:
            vals = []
            for N in Ns:
                recs = ps[f'A|{N}|{pol}']
                if metric == 'lat':
                    vals.append([min(x['lat'], 60.0) for x in recs])
                else:
                    vals.append([100 * x['miss'] for x in recs])
            kw = dict(ls=':', mfc='white') if pol == 'rbr_b100' else {}
            line(ax, bs, key, xs, vals, label=lab, record=r, **kw)
        if metric == 'lat':
            lm = [np.mean([x['lat'] for x in ps[f'A|{N}|local']]) for N in Ns]
            ax.plot(xs, lm, color=INK2, ls='--', lw=1, label=LEGEND['local'], zorder=1)
            ax.set_ylim(0, 63)
        ax.set_xlabel(T['xlabel'])
        ax.set_ylabel(yl)
        title(ax, ttl)
        ax.set_xticks(xs)
        ax.set_xticklabels([T['ticks'][N] for N in Ns])
    for metric in rec:
        for lab in rec[metric]:
            for pt in rec[metric][lab]:
                pt['N'] = Ns[int(pt['x'])]
    u = [np.mean([x['unstable'] for x in ps[f'A|{N}|rbr_b100']]) for N in Ns]
    uu = [(N, v) for N, v in zip(Ns, u) if v > 0]
    note = (', '.join(f'{v:.0%}' for _, v in uu[:-1]) + f' and {uu[-1][1]:.0%} of runs at N = '
            + ', '.join(str(N) for N, _ in uu[:-1]) + f' and {uu[-1][0]}')
    if T['footer']:
        fig.text(0.01, 0.0, T['footer'].format(note=note), fontsize=6.3, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    # EDIT: legend position (shared legend under both panels, 3 columns)
    fig.legend(h, lab, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.075))
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    VALUES['R5'] = rec
    save(fig, 'fig_r5_asynchronous')


def fig_r5b():
    """Asynchronous arrivals with a persistent on-board processor per vehicle (E5b). Manuscript Fig. 9."""
    T = TEXT['R5b']
    a = load('res_async_local.json')
    ps = a['per_seed']
    Ns = a['fleets']
    xs = list(range(len(Ns)))
    series = [('plan+own', 'plan', T['plan_own'], {}),
              ('plan', 'plan', T['plan_e5'], dict(ls=':', mfc='white')),
              ('gipa_slot+own', 'gipa', T['gipa_own'], {}),
              ('la', 'la', None, {}),
              ('jsq_live', 'greedy', T['jsq'], {}),
              ('rbr_b20', 'rbr', T['rbr20'], {})]
    rec = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, metric, ttl, yl in [(axes[0], 'lat', T['title_a'], T['ylabel_a']),
                                (axes[1], 'miss', T['title_b'], T['ylabel_b'])]:
        bs = Bootstrap()
        r = rec.setdefault(metric, {})
        for pol, key, lab, kw in series:
            vals = []
            for N in Ns:
                recs = ps[f'fifo|A|{N}|{pol}']
                if metric == 'lat':
                    vals.append([min(x['lat'], 60.0) for x in recs])
                else:
                    vals.append([100 * x['miss'] for x in recs])
            line(ax, bs, key, xs, vals, label=lab, record=r, **kw)
        if metric == 'lat':
            ax.set_ylim(0, 63)
        ax.set_xlabel(T['xlabel'])
        ax.set_ylabel(yl)
        title(ax, ttl)
        ax.set_xticks(xs)
        ax.set_xticklabels([T['ticks'][N] for N in Ns])
    for metric in rec:
        for lab in rec[metric]:
            for pt in rec[metric][lab]:
                pt['N'] = Ns[int(pt['x'])]
    unst = [(pol, N, np.mean([x['unstable'] for x in ps[f'fifo|A|{N}|{pol}']])) for pol, *_ in series for N in Ns]
    unst = [u for u in unst if u[2] > 0]
    note = ''
    if unst:
        names = {'plan+own': T['plan_own'], 'plan': T['plan_e5'], 'gipa_slot+own': T['gipa_own'],
                 'la': LEGEND['la'], 'jsq_live': T['jsq'], 'rbr_b20': T['rbr20']}
        note = '\nUnbounded backlog: ' + '; '.join(f'{names[p]}, {v:.0%} of runs at N = {N}' for p, N, v in unst) + '.'
    if T['footer']:
        fig.text(0.01, 0.0, T['footer'].format(note=note), fontsize=6.3, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    # EDIT: legend position (shared legend under both panels, 3 columns)
    fig.legend(h, lab, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0 if not T['footer'] else 0.075))
    fig.tight_layout(rect=(0, 0.16 if not T['footer'] else 0.22, 1, 1))
    VALUES['R5b'] = rec
    save(fig, 'fig_r5b_async_local_queue')


def fig_r6():
    """Learning automata: learning curve and density change (E6). Manuscript Fig. 9."""
    T = TEXT['R6']
    la = load('res_la.json')
    rec = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.5]))
    ax = axes[0]
    curve = la['main']['A|affine|500']['160']['best']['curve']
    x = (np.arange(len(curve)) + 1) * 25
    ax.plot(x, curve, color=STYLE['la']['color'], marker=STYLE['la']['marker'], ms=4, label=T['la'])
    ref_line(ax, 18.38, T['ref_gipa'], STYLE['gipa']['color'])
    ref_line(ax, 16.69, T['ref_plan'], STYLE['plan']['color'])
    ax.axvline(25, color=INK2, lw=0.8, ls=':')
    ax.annotate(T['warmup'], (25, 19.6), xytext=(5, 0), textcoords='offset points', fontsize=6.5, color=INK2)
    ax.set_xlabel(T['xlabel_a'])
    ax.set_ylabel(T['ylabel_a'])
    title(ax, T['title_a'])
    ax.set_ylim(16.2, 21)
    rec['curve'] = dict(slot=x.tolist(), lat=curve)
    ax = axes[1]
    dc = la['density_change']['policies']
    sched = la['density_change']['schedule']
    t = np.arange(len(sched))
    ax.axvspan(500, 1000, color='#f0efec', zorder=0)
    for xp, txt in zip([250, 750, 1250], T['phases']):
        ax.annotate(txt, (xp, 14.0), ha='center', fontsize=7, color=INK2)
    k = 10
    for name, key, ls in [('plan_fresh', 'plan', '-'), ('gipa_fresh', 'gipa', '-'),
                          ('gipa_stale', 'gipa', ':'), ('la_lrep', 'la', '-')]:
        y = np.convolve(dc[name]['trace']['lat'], np.ones(k) / k, mode='same')
        ax.plot(t[k:-k], y[k:-k], color=STYLE[key]['color'], ls=ls, lw=1.3, label=T[name])
        rec[name] = dc[name]['phases']
    ax.set_ylim(13.5, 27.5)
    ax.set_xlabel(T['xlabel_b'])
    ax.set_ylabel(T['ylabel_b'])
    title(ax, T['title_b'])
    ax.legend(loc='upper right', fontsize=6.5)  # EDIT: legend position, panel (b)
    fig.tight_layout()
    VALUES['R6'] = rec
    save(fig, 'fig_r6_learning_automata')


def fig_r7():
    """Anti-herding with one RSU 30% faster (E9). Manuscript Fig. 7."""
    T = TEXT['R7']
    a = load('res_asym.json')
    ps = a['per_seed']
    key = 'fast1|A|affine'
    pols = [('greedy', 'greedy', LEGEND['greedy']), ('msoft_prior|capacity', 'msoft', LEGEND['msoft_prior']),
            ('gipa_nodiff|capacity', 'nodiff', LEGEND['nodiff']), ('rbr', 'rbr', LEGEND['rbr']),
            ('gipa|uniform', 'gipa', LEGEND['gipa']), ('la', 'la', LEGEND['la']),
            ('plan', 'plan', LEGEND['plan'])]
    rec = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    bs = Bootstrap()
    y = np.arange(len(pols))[::-1]
    ms, los, his = [], [], []
    for pol, k, _ in pols:
        m, lo, hi = ci(bs, ps[f'{key}|80|{pol}']['herd_fast'])
        ms.append(m)
        los.append(lo)
        his.append(hi)
    ms, los, his = map(np.array, (ms, los, his))
    ax.barh(y, ms, 0.62, color=[STYLE[k]['color'] for _, k, _ in pols], zorder=2)
    ax.errorbar(ms, y, xerr=[ms - los, his - ms], fmt='none', ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
    for yy, m in zip(y, ms):
        ax.annotate(f'{m:.2f}', (m, yy), xytext=(3, 0), textcoords='offset points', va='center', fontsize=7, color=INK)
    ax.axvline(1, color=INK2, ls='--', lw=1)
    ax.annotate(T['fair'], (1, y[-1] - 0.75), xytext=(3, 0), textcoords='offset points', fontsize=6.3,
                color=INK2, va='center')
    ax.axvline(2.4, color=INK2, ls=':', lw=1)
    ax.annotate(T['all_fast'], (2.4, y[-1] - 0.75), xytext=(3, 0), textcoords='offset points', fontsize=6.3,
                color=INK2, va='center')
    ax.set_ylim(y[-1] - 1.2, y[0] + 0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, _, lab in pols])  # method names act as the legend
    ax.set_xlim(0, 2.95)
    ax.set_xlabel(T['xlabel_a'])
    title(ax, T['title_a'])
    rec['herd_N80'] = {lab: dict(mean=float(m), ci95=[float(lo), float(hi)]) for (_, _, lab), m, lo, hi in zip(pols, ms, los, his)}
    ax = axes[1]
    bs = Bootstrap()
    r = rec.setdefault('lat', {})
    for pol, k, lab in [('plan', 'plan', None), ('la', 'la', None), ('gipa|uniform', 'gipa', None),
                        ('gipa|capacity', 'gipa', T['gipa_cap']),
                        ('gipa_nodiff|capacity', 'nodiff', None)]:
        kw = dict(ls=':', mfc='white') if pol == 'gipa|capacity' else {}
        line(ax, bs, k, DENS, [ps[f'{key}|{N}|{pol}']['lat'] for N in DENS], label=lab, record=r, **kw)
    lm = [np.mean(ps[f'{key}|{N}|local']['lat']) for N in DENS]
    ax.plot(DENS, lm, color=INK2, ls='--', lw=1, label=LEGEND['local'], zorder=1)
    ax.set_xticks(DENS)
    ax.set_xlabel(T['xlabel_b'])
    ax.set_ylabel(T['ylabel_b'])
    ax.set_ylim(13, 26.5)
    title(ax, T['title_b'])
    ax.legend(loc='upper left', fontsize=6.3, ncol=2)  # EDIT: legend position, panel (b)
    fig.tight_layout()
    VALUES['R7'] = rec
    save(fig, 'fig_r7_asymmetric_rsus')


def fig_r8(path=None):
    """Decision time and runtime payload (E8). Manuscript Fig. 10."""
    T = TEXT['R8']
    c = load(path or 'res_complexity.json')
    fl = c['fleet']['160']
    sv = c['single_vehicle']
    comm = c['communication']['rows']
    rows = [(LEGEND['gipa'], 'gipa', fl['gipa']['median_us'], sv['gipa_scalar']['median_us'], comm['gipa']['Bps']),
            (LEGEND['plan'], 'plan', fl['plan']['median_us'], sv['plan']['median_us'], comm['plan']['Bps']),
            (LEGEND['la'], 'la', fl['la_lri']['median_us'], sv['la_lri']['median_us'], comm['la']['Bps']),
            (T['mappo'], 'mappo', fl['mappo']['median_us'], sv['mappo']['median_us'], comm['mappo (base)']['Bps']),
            (LEGEND['ql'], 'ql', fl['ql']['median_us'], sv['ql']['median_us'], comm['ql']['Bps']),
            (LEGEND['rbr'], 'rbr', fl['rbr']['median_us'], None, comm['rbr']['Bps']),
            (LEGEND['price'], 'greedy', fl['price']['median_us'], None, comm['price']['Bps'])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    y = np.arange(len(rows))[::-1]
    ax = axes[0]
    ax.barh(y, [r[2] for r in rows], 0.6, color=[STYLE[r[1]]['color'] for r in rows], zorder=2)
    for yy, r in zip(y, rows):
        note = T['notes'].get(r[1], '')
        txt = f'{r[2]:.0f} μs  ' + (T['per_vehicle'].format(v=r[3]) if r[3] is not None else f'({note})')
        ax.annotate(txt, (r[2], yy), xytext=(3, 0), textcoords='offset points', va='center', fontsize=6.5, color=INK,
                    bbox=dict(fc='white', ec='none', pad=0.6), zorder=4)
    ax.set_xscale('log')
    ax.set_xlim(5, 1e6)
    ax.axvline(20000, color=INK2, ls='--', lw=1)
    ax.annotate(T['slot'], (20000, 1.0), xycoords=('data', 'axes fraction'), xytext=(0, 2), textcoords='offset points',
                fontsize=6.3, color=INK2, ha='center', va='bottom')
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])  # method names act as the legend
    ax.set_xlabel(T['xlabel_a'])
    title(ax, T['title_a'])
    ax = axes[1]
    vals = [max(r[4], 0.5) for r in rows]
    ax.barh(y, vals, 0.6, color=[STYLE[r[1]]['color'] for r in rows], zorder=2)
    for yy, r, v in zip(y, rows, vals):
        lab = T['none'] if r[4] == 0 else (f'{r[4] / 1e6:.1f} MB/s' if r[4] >= 1e6 else f'{r[4]:.0f} B/s')
        ax.annotate(lab, (v, yy), xytext=(3, 0), textcoords='offset points', va='center', fontsize=6.5, color=INK)
    ax.set_xscale('log')
    ax.set_xlim(0.4, 1e8)
    ax.set_xlabel(T['xlabel_b'])
    title(ax, T['title_b'])
    hw = c['hardware']
    full = hw.get('full', {})
    cpu = hw['cpu'].replace('(R)', '')
    parts = [cpu]
    if full:
        vm = f", {full['processor']['logical_cpus']} vCPU ({full['hypervisor_vendor']} virtual machine)" if full.get('virtualised') else ''
        parts = [cpu + vm, full['os'], f"CPython {full['python']['version']}", f"NumPy {full['packages']['numpy']}"]
    reps = c.get('repeats', 1)
    tail = f'median of {reps} runs, each the median of 10⁵ decisions' if reps > 1 else 'median of 10⁵ decisions'
    if T['footer']:
        fig.text(0.01, -0.02, T['footer'].format(machine='; '.join(parts), tail=tail), fontsize=6.3, color=INK2, va='top')
    fig.tight_layout()
    VALUES['R8'] = dict(hardware=c['hardware'], rows=[dict(policy=r[0], fleet_us=r[2], vehicle_us=r[3], payload_Bps=r[4]) for r in rows])
    save(fig, 'fig_r8_decision_cost')


def fig_r9():
    """Heterogeneous tasks: amplification on the plan versus per-task ranking (E12). Manuscript Fig. 11."""
    T = TEXT['R9']
    h = load('res_hetero.json')
    ps = h['per_seed']
    sc = 'het50'
    series = [('plan', 'plan', T['plan'], {}),
              ('plan_rank', 'plan', T['plan_rank'], dict(ls='--', mfc='white')),
              ('gipa', 'gipa', None, {}),
              ('gipa_plan', 'plan', T['gipa_plan'], dict(ls=':', marker='^')),
              ('la', 'la', None, {})]
    rec = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, metric, ttl, yl, scale in [(axes[0], 'lat', T['title_a'], T['ylabel_a'], 1),
                                       (axes[1], 'miss', T['title_b'], T['ylabel_b'], 100)]:
        bs = Bootstrap()
        r = rec.setdefault(metric, {})
        for pol, key, lab, kw in series:
            line(ax, bs, key, DENS, [[scale * x[metric] for x in ps[f'{sc}|{N}|{pol}']] for N in DENS],
                 label=lab, record=r, **kw)
        if metric == 'lat':
            lm = [np.mean([x['lat'] for x in ps[f'{sc}|{N}|local']]) for N in DENS]
        else:
            lm = [100 * np.mean([x['miss'] for x in ps[f'{sc}|{N}|local']]) for N in DENS]
        ax.plot(DENS, lm, color=INK2, ls='--', lw=1, label=LEGEND['local'], zorder=1)
        ax.set_xticks(DENS)
        ax.set_xlabel(T['xlabel'])
        ax.set_ylabel(yl)
        title(ax, ttl)
    h_, lab = axes[0].get_legend_handles_labels()
    # EDIT: legend position (shared legend under both panels, 3 columns)
    fig.legend(h_, lab, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    VALUES['R9'] = rec
    save(fig, 'fig_r9_heterogeneous_tasks')


FIGS = {1: fig_r1, 2: fig_r2, 3: fig_r3, 4: fig_r4, 5: fig_r5, 6: fig_r6, 7: fig_r7, 8: fig_r8, 9: fig_r9, 10: fig_r5b}  # 10 = R5b (E5b)


def main():
    global PANEL_TITLES
    ap = argparse.ArgumentParser(description='Revision summary figures')
    ap.add_argument('--figure', type=int, choices=sorted(FIGS), help='draw one figure (R1-R9; 10 = R5b) only')
    ap.add_argument('--complexity', default=None, help='complexity result file name in results/ (e.g. res_complexity_laptop.json)')
    ap.add_argument('--titles', choices=['full', 'letter', 'none'], default=None,
                    help='panel titles: full description, letter only, or none (default: PANEL_TITLES)')
    args = ap.parse_args()
    if args.titles:
        PANEL_TITLES = args.titles
    todo = [args.figure] if args.figure else sorted(FIGS)
    vpath = RES / 'res_revision_figures.json'
    if vpath.exists():
        VALUES.update(json.load(open(vpath)))
    for k in todo:
        if k == 8:
            fig_r8(args.complexity)
        else:
            FIGS[k]()
    with open(vpath, 'w') as fh:
        json.dump(VALUES, fh, indent=1)
    print('wrote', vpath)


if __name__ == '__main__':
    main()
