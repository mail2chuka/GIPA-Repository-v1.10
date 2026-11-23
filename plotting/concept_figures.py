"""
concept_figures.py
==================
Explanatory (non-result) figures for the first revision of ITS-2026-08-0579.
Every quantity drawn is computed from the model of the paper (Sections III-V);
nothing is read from simulation results.

C1  fig_c1_system_architecture   Vehicles, RSUs with processor sharing, the prior
                                 broadcast and the per-vehicle decision
                                 (replacement for Fig. 1)
C2  fig_c2_gipa_framework        Two-timescale GIPA framework with equation
                                 numbers (replacement for Fig. 2)
C3  fig_c3_herding               Why randomisation matters: a deterministic rule on
                                 shared information versus independent sampling
                                 from the plan (N = 160)
C4  fig_c4_equilibrium_optimum   User equilibrium versus system optimum on one RSU:
                                 own latency, marginal cost, saving (Props. 1-2)
C5  fig_c5_operator_walkthrough  How Stage 1 updates the probabilities, round by
                                 round, for both priors (Eqs. 15-23)
C6  fig_c6_deadline_margin       Load distribution under independent sampling and
                                 the chance-constrained margin (Prop. 3, Regime B)
C7  fig_c7_processor_sharing     Processor sharing on one RSU: exact Eq. (5)
                                 versus the simpler n_j c_i / F_j

Outputs: figures/concept/<stem>.{png (300 dpi), pdf, svg, tif (600 dpi, LZW)}.

WHERE TO EDIT
-------------
All wording is in the TEXT block marked "EDIT HERE"; numerical settings of each
figure (fleet size, seeds, illustrative task sizes) are in PARAMS. Rerun the
script after editing.

Run (from the repository root):
    python plotting/concept_figures.py            # all
    python plotting/concept_figures.py --figure 4 # one
"""

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import binom  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures' / 'concept'

# ======================================================================
# EDIT HERE: wording
# ======================================================================
TEXT = {
    'C1': dict(
        prior_box='Stage 0 / Stage 2: prior $\\phi_k$ from the anticipated fleet size, class mix and latency model\n'
                  'broadcast through the RSUs every $T_{\\mathrm{up}}$ slots (60 B/s); no per-slot messages',
        rsu='RSU {j}', server='RSU {j}: edge server $F_{{{j}}}$\ntasks share the processor (5)',
        vehicle_box='Vehicle $i$, for every task (Stage 1)\n'
                    '1. mask: feasible actions $\\mathcal{F}_i(t)$ (8)\n'
                    '2. initialise from the prior\n'
                    '3. amplify and diffuse, $K$ rounds\n'
                    '4. sample $a_i\\sim p_i^{(K)}$: local or one RSU',
        legend_local='Local execution', legend_off='Offload to RSU', legend_prior='Prior broadcast (slow)',
        legend_cov='RSU coverage'),
    'C2': dict(
        slow='Slow timescale: every $T_{\\mathrm{up}}$ slots',
        s0='Stage 0: the prior (one of two)',
        inputs='Anticipated\nfleet size $\\bar N$\nclass mix $\\pi_k$\nlatency model\n(7), (5) or\nasynchronous',
        heur_t='Heuristic prior (11)', heur_b='expected congestion if the fleet\nspreads evenly: describes a load',
        heur_val='{c:.1f} ms',
        plan_t='Planned prior: P1 (12)–(14)', plan_b='share of each class sent to each\naction: prescribes the optimum',
        classes=['Safety', 'Background'],
        s2_t='Stage 2: refresh',
        s2_b='RSUs count the vehicles\nassociated with them\n$\\bar N \\leftarrow (1-\\alpha)\\bar N+\\alpha\\,$count\nre-run Stage 0, broadcast\n60 B/s planned, 90 B/s heuristic',
        refresh='new $\\bar N$, $\\pi_k$',
        broadcast='prior broadcast through the RSUs\n(no messages between vehicles)', either='or',
        plan_each='{v:.3f} to each RSU',
        counts='association\ncounts',
        fast='Stage 1: every task, on every vehicle, no communication',
        example='Example: a background task; RSU 3 is out of reach; the link to RSU 1 is 0.5 ms faster than average',
        mask_t='1  Mask (8)',
        mask_rows=['Local', 'RSU 1', 'RSU 2', 'RSU 3'],
        mask_why=['', '', '', 'out of reach'],
        mask_b='Remove actions out of\ncoverage, past the dwell\ntime or predicted to miss\nthe deadline (guard\nband $\\gamma$). Nothing\nfeasible: run locally.',
        cols=['2  Initialise', '3  Amplify and diffuse, $K$ rounds', '4  Sample', 'Effect'],
        rows=['GIPA with\nheuristic prior', 'GIPA with\nplanned prior', 'Planned prior,\nper-task\nranking (24)'],
        a_init='softmax of\n$q-\\bar c$ (15), (17)', a_amp='pulled towards\nuniform; floor $\\lambda u$\n(21), (22)',
        b_init='the plan $\\phi_B$\n(18)', b_amp='tilted by the\nadvantage $q-\\bar q_B$\n(16), (21), (22)',
        c_rank='offload if $\\beta_i$ (24) > $\\tau_B$: the top {r:.0%} of\nbackground tasks (the planned share)',
        c_task='this task', c_tau='$\\tau_B$', c_x='predicted benefit of offloading',
        c_pick='RSU drawn from $\\rho_{{j\\mid B}}$\nover reachable RSUs',
        draw='$u\\sim U(0,1)$', chosen='→ {a}',
        a_eff='spreads the load,\nbut stops where\noffloading no longer\npays the vehicle\n(user equilibrium)',
        b_eff='an average task\nsamples the plan;\nthis one leans\nto RSU 1',
        c_eff='keeps the planned\nshare exactly and\nfills it with the\ntasks that gain\nmost'),
    'C3': dict(
        title_a=('(a)', 'Deterministic choice on shared information'),
        title_b=('(b)', 'GIPA: independent sampling from the plan'),
        cols=['Local', 'RSU 1', 'RSU 2', 'RSU 3'],
        cap='deadline capacity\n(59 background tasks)',
        note_a='every task at RSU 1:\n{lat:.0f} ms each, deadline 40 ms',
        note_b='{off} offloaded, {per} per RSU on average\nlargest load {mx}; mean latency {lat:.1f} ms',
        unit='one square = one background task (N = {N} vehicles, {nb} background tasks)'),
    'C4': dict(
        title_a=('(a)', 'One RSU: own latency versus marginal cost'),
        title_b=('(b)', 'Fleet mean latency, N = 160'),
        local='local latency $L_B$ = 22 ms',
        own='latency of the offloaded task\n$e_B+\\xi(n-1)$ (what a vehicle sees)',
        marg='marginal cost to the fleet\n$e_B+\\xi(2n-1)$',
        ue='user equilibrium\n$n^{{\\mathrm{{UE}}}}$ = {nue:.2f}',
        so='system optimum\n$n^*$ = {ns:.2f}',
        saving='saving at this RSU\n= shaded area\n= $\\xi n^{{*2}}$ = {sv:.0f} ms/slot',
        xlabel_a='Background tasks at the RSU, $n$', ylabel_a='Latency (ms)',
        xlabel_b='Background tasks offloaded per RSU', ylabel_b='Mean latency over all tasks (ms)',
        curve='expected, coordinated', pts={'heur': 'GIPA, heuristic prior (simulated)',
                                            'plan': 'GIPA, planned prior (simulated)',
                                            'local': 'All-local (simulated)'}),
    'C5': dict(
        title_a=('(a)', 'Heuristic prior'),
        title_b=('(b)', 'Planned prior, identical tasks'),
        title_c=('(c)', 'Planned prior, faster link to RSU 1'),
        actions=['Local', 'RSU 1', 'RSU 2', 'RSU 3'],
        floor='floor $\\lambda_B u_a$',
        xlabel='Round $r$', ylabel='Probability $p^{(r)}_{i,a}$',
        note_a='adjusted scores favour local;\ndiffusion pulls towards the\nuniform target (RSUs 1–3 coincide)',
        note_b='adjusted scores are all zero,\nso the operator returns\nthe plan (RSUs 1–3 coincide)',
        note_c='amplification moves\nprobability to RSU 1\n($\\beta$ = {beta}, $\\lambda$ = 0;\nRSUs 2 and 3 coincide)'),
    'C6': dict(
        title=('', 'Safety task in Regime B, N = 80: tasks sharing its RSU'),
        z0='plan without margin ($z$ = 0)', z2='plan with margin ($z$ = 2)',
        cap='deadline missed to the right\n(17 or more other tasks)',
        tail='miss: {p:.1%} predicted,\n{m:.1%} measured',
        xlabel='Other tasks at the same RSU, $m$', ylabel='Probability'),
    'C7': dict(
        title_a=('(a)', 'Processor share over time'),
        title_b=('(b)', 'Completion time of each task'),
        task='Task {k}\n{c:g} Mcycles',
        exact='exact, Eq. (5)', naive='$n_jc_i/F_j$',
        xlabel_a='Time (ms)', ylabel_a='Share of the processor',
        xlabel_b='Completion time (ms)'),
}

PARAMS = {
    'N': 160, 'piS': 0.3, 'M': 3, 'xi': 0.55, 'eB': 8.0, 'LB': 22.0, 'eS': 3.0, 'LS': 10.0,
    'phi_B': [0.646, 0.118, 0.118, 0.118], 'seed': 43,
    'op': dict(tau=0.4, beta=0.4, nu=0.3, theta0=1.5, gamma_theta=0.6, K=4, eta=1.0, lamB=0.6),
    'C5_scores': [22.0, 7.8, 8.0, 8.3], 'C5_beta_c': 0.8,
    'C2_fast': 0.5, 'C2_beta': 0.4, 'C2_lam': 0.0, 'C2_u': (0.55, 0.70, 0.30),   # example task, tuned beta/lambda, draws
    'C6_q': {'z0': 0.2167, 'z2': 0.1428}, 'C6_measured': {'z0': 0.544, 'z2': 0.042}, 'C6_N': 80, 'C6_cap': 17,
    'C7_tasks': [2.0, 4.0, 4.0, 8.0], 'C7_F': 1.0,
    'measured': {'local': (0.0, 18.38), 'heur': (25.8, 18.38), 'plan': (13.2, 16.69)},
}
FORMATS = ['png', 'pdf', 'svg', 'tif']
PNG_DPI, TIFF_DPI = 300, 600
# ======================================================================
# end of EDIT HERE
# ======================================================================

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
ACT = ['#52514e', '#1baf7a', '#6250d6', '#e87ba4']      # local, RSU 1, RSU 2, RSU 3
FILL = {'blue': '#e8f1fc', 'green': '#e6f6ef', 'orange': '#fdf0e7', 'grey': '#f2f1ee'}

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
})


def title(ax, t):
    ax.set_title(' '.join(x for x in t if x), loc='left')


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        if ext == 'tif':
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=TIFF_DPI, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            Image.open(buf).convert('RGB').save(OUT / f'{stem}.tif', format='TIFF', compression='tiff_lzw',
                                                dpi=(TIFF_DPI, TIFF_DPI))
        else:
            fig.savefig(OUT / f'{stem}.{ext}', dpi=PNG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', OUT / stem)


def box(ax, x, y, w, h, text='', fc='white', ec=INK2, fs=7, lw=0.9, r=0.05, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.02,rounding_size={r}', fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.3, **kw)


def arrow(ax, p, q, color=INK2, ls='-', lw=1.0, rad=0.0, ms=8, style='-|>'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=1, shrinkB=1))


# ----------------------------------------------------------------------------- C1
def car(ax, x, y, color, w=0.62, h=0.26):
    ax.add_patch(FancyBboxPatch((x - w / 2, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.08', fc=color, ec='white',
                                lw=0.6, zorder=4))
    ax.add_patch(FancyBboxPatch((x - w * 0.25, y + h * 0.75), w * 0.5, h * 0.45,
                                boxstyle='round,pad=0.0,rounding_size=0.06', fc=color, ec='white', lw=0.6, zorder=4))
    for dx in (-w * 0.28, w * 0.28):
        ax.add_patch(Circle((x + dx, y), 0.06, fc=INK, ec='white', lw=0.4, zorder=5))


def fig_c1():
    T = TEXT['C1']
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    ax.set_xlim(0, 14)
    ax.set_ylim(-1.85, 8.0)
    ax.axis('off')
    # road
    ax.add_patch(Rectangle((0, 0.6), 14, 1.5, fc='#d9d8d4', ec='none', zorder=1))
    ax.plot([0, 14], [1.35, 1.35], color='white', lw=1.4, ls=(0, (6, 5)), zorder=2)
    rx = [2.3, 7.0, 11.7]
    for j, x in enumerate(rx):
        c = ACT[j + 1]
        ax.add_patch(Wedge((x, 3.55), 3.1, 205, 335, fc=c, alpha=0.05, ec=c, lw=0.6, ls='--', zorder=1))
        # tower
        ax.add_patch(Polygon([[x - 0.18, 2.3], [x + 0.18, 2.3], [x + 0.05, 3.55], [x - 0.05, 3.55]], fc=INK2, ec='none',
                             zorder=3))
        for rr in (0.25, 0.42):
            ax.add_patch(Wedge((x, 3.6), rr, 30, 150, width=0.03, fc=INK2, ec='none', zorder=3))
        # edge server with processor sharing
        box(ax, x - 1.85, 4.08, 3.7, 1.55, fc=FILL['grey'], ec=c, lw=1.1, r=0.08)
        ax.text(x, 5.5, T['server'].format(j=j + 1), ha='center', va='top', fontsize=6.4, color=INK, linespacing=1.25)
        lens = [0.7, 1.3, 1.9] if j != 1 else [0.6, 1.6]
        for k, L in enumerate(lens):
            ax.add_patch(Rectangle((x - 1.45, 4.2 + k * 0.2), L, 0.14, fc=c, ec='none', alpha=0.85))
    # prior box and broadcast
    box(ax, 0.6, 6.55, 12.8, 1.15, T['prior_box'], fc=FILL['blue'], ec=BLUE, fs=7, lw=1.1)
    for x in rx:
        arrow(ax, (x, 6.55), (x, 5.66), color=BLUE, ls=(0, (2, 2)), lw=1.0)
    # cars and their choices
    cars = [(0.8, 0.72, 0), (1.9, 1.45, 1), (3.2, 0.72, 1), (4.3, 1.45, 0), (5.5, 0.72, 2), (6.6, 1.45, 0),
            (8.0, 0.72, 2), (9.1, 1.45, 3), (10.3, 0.72, 0), (11.5, 1.45, 3), (12.8, 0.72, 0)]
    for x, y, a in cars:
        car(ax, x, y, ACT[a])
        if a == 0:
            ax.add_patch(FancyArrowPatch((x + 0.18, y + 0.62), (x - 0.18, y + 0.62), connectionstyle='arc3,rad=-1.6',
                                         arrowstyle='-|>', mutation_scale=6, color=ACT[0], lw=0.8, zorder=6))
        else:
            arrow(ax, (x, y + 0.45), (rx[a - 1], 2.35), color=ACT[a], ls=(0, (3, 2)), lw=0.9, rad=0.0)
    # per-vehicle decision (below the road) and legend
    box(ax, 0.15, -1.75, 5.4, 2.05, T['vehicle_box'], fc=FILL['green'], ec=GREEN, fs=6.4, lw=1.0)
    arrow(ax, (2.9, 0.3), (3.2, 0.68), color=GREEN, lw=0.9)
    hs = [plt.Line2D([], [], color=ACT[0], marker=r'$\circlearrowleft$', ls='none', ms=8, label=T['legend_local']),
          plt.Line2D([], [], color=ACT[1], ls=(0, (3, 2)), lw=1.2, label=T['legend_off'] + ' 1'),
          plt.Line2D([], [], color=ACT[2], ls=(0, (3, 2)), lw=1.2, label=T['legend_off'] + ' 2'),
          plt.Line2D([], [], color=ACT[3], ls=(0, (3, 2)), lw=1.2, label=T['legend_off'] + ' 3'),
          plt.Line2D([], [], color=BLUE, ls=(0, (2, 2)), lw=1.2, label=T['legend_prior']),
          Rectangle((0, 0), 1, 1, fc=ACT[1], alpha=0.12, ec=ACT[1], ls='--', lw=0.6, label=T['legend_cov'])]
    ax.legend(handles=hs, loc='center left', bbox_to_anchor=(0.42, 0.08), ncol=2, fontsize=6.6, handlelength=2.4,
              columnspacing=1.6)
    save(fig, 'fig_c1_system_architecture')


# ----------------------------------------------------------------------------- C2
def mini_bars(fig, rect, p, before=None, floor=None, note=None, note_y=0.95):
    """Small action-PMF bar chart: Local, RSU 1, RSU 2, RSU 3 (RSU 3 unreachable in the example)."""
    ax = fig.add_axes(rect)
    x = np.arange(4)
    pp = np.r_[p, 0.0]
    if before is not None:
        ax.bar(x[:3], before, 0.62, fc='none', ec=INK2, lw=0.6, ls=(0, (2, 1.5)), zorder=3)
    ax.bar(x[:3], pp[:3], 0.62, color=ACT[:3], zorder=2)
    for xi, v in zip(x[:3], pp[:3]):
        ax.text(xi, v + 0.04, f'{v:.2f}', ha='center', va='bottom', fontsize=5.4, color=INK)
    ax.text(3, 0.03, '×', ha='center', va='bottom', fontsize=7, color=INK2)
    if floor is not None:
        ax.axhline(floor, color=INK2, lw=0.7, ls=':', zorder=4)
    if note:
        ax.text(1.0, note_y, note, transform=ax.transAxes, ha='right', va='top', fontsize=5.6, color=INK2,
                linespacing=1.15)
    ax.set_ylim(0, 1.22)
    ax.set_xlim(-0.55, 3.55)
    ax.set_xticks(x)
    ax.set_xticklabels(['L', '1', '2', '3'], fontsize=6)
    ax.tick_params(axis='x', length=0, pad=1.5)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    return ax


def strip(fig, rect, p, u, T, labels=('Local', 'RSU 1', 'RSU 2')):
    """Sampling as a draw on a unit strip split by the PMF; the segment that contains u is chosen."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    edges = np.r_[0, np.cumsum(p)]
    pick = int(np.searchsorted(edges, u, side='right') - 1)
    for a in range(len(p)):
        if p[a] <= 0:
            continue
        ax.add_patch(Rectangle((edges[a], 0.38), p[a], 0.26, fc=ACT[a], ec='white', lw=1.0,
                               alpha=1.0 if a == pick else 0.45))
    ax.add_patch(Polygon([[u - 0.035, 0.80], [u + 0.035, 0.80], [u, 0.66]], closed=True, fc=INK, ec='none'))
    ax.text(u, 0.83, T['draw'], ha='center', va='bottom', fontsize=5.8, color=INK)
    ax.text(0.5, 0.30, T['chosen'].format(a=labels[pick]), ha='center', va='top', fontsize=6.6, color=INK,
            weight='bold')
    return pick


def fig_c2():
    T = TEXT['C2']
    P = PARAMS
    o = P['op']
    W, H = 7.2, 6.2
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis('off')

    def fbox(x, y, w, h, **kw):
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.008', **kw))

    # example task: background class, RSU 3 unreachable, faster link to RSU 1
    LB, eB, xi, N, M = P['LB'], P['eB'], P['xi'], P['N'], P['M']
    q = -np.array([LB, eB - P['C2_fast'], eB])
    cbar = xi * max(N / (M + 1) - 1, 0)
    phi = np.array(P['phi_B'])
    phi_r = np.r_[phi[0], phi[1:3] / phi[1:3].sum() * phi[1:].sum()]      # unreachable mass moved, (18)
    kw = dict(tau=o['tau'], nu=o['nu'], theta0=o['theta0'], gamma_theta=o['gamma_theta'], K=o['K'])
    hA = operator(q, o['eta'] * np.array([0, cbar, cbar]), u=np.ones(3) / 3, beta=o['beta'], lam=o['lamB'], **kw)
    hB = operator(q, -np.array([LB, eB, eB]), p0=phi_r, u=phi_r, beta=P['C2_beta'], lam=P['C2_lam'], **kw)
    r_off = 1 - phi[0]

    # ---------------- slow band
    fbox(0.012, 0.712, 0.976, 0.280, fc='#f7fafe', ec=BLUE, lw=0.8, ls=(0, (4, 3)))
    bg.text(0.025, 0.978, T['slow'], fontsize=7.4, color=BLUE, va='top', style='italic')
    bg.text(0.43, 0.978, T['s0'], fontsize=8, va='top', ha='center', weight='bold')
    fbox(0.025, 0.735, 0.125, 0.195, fc='white', ec=INK2, lw=0.8)
    bg.text(0.0875, 0.8325, T['inputs'], ha='center', va='center', fontsize=6.4, linespacing=1.3)
    # heuristic prior option
    fbox(0.180, 0.718, 0.225, 0.232, fc='white', ec=ORANGE, lw=1.3)
    bg.text(0.2925, 0.942, T['heur_t'], ha='center', va='top', fontsize=7.2, weight='bold', color=ORANGE)
    ax = fig.add_axes([0.203, 0.790, 0.18, 0.115])
    vals = np.r_[0, [cbar] * 3]
    ax.bar(range(4), vals, 0.6, color=ACT)
    ax.text(2, cbar + 1.2, T['heur_val'].format(c=cbar) + ' at each RSU', ha='center', va='bottom', fontsize=5.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['L', '1', '2', '3'], fontsize=6)
    ax.set_ylim(0, cbar * 1.45)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', length=0, pad=1.5)
    bg.text(0.2925, 0.727, T['heur_b'], ha='center', va='bottom', fontsize=5.9, color=INK2, linespacing=1.2)
    # planned prior option
    fbox(0.455, 0.718, 0.225, 0.232, fc='white', ec=BLUE, lw=1.6)
    bg.text(0.5675, 0.942, T['plan_t'], ha='center', va='top', fontsize=7.2, weight='bold', color=BLUE)
    ax = fig.add_axes([0.530, 0.795, 0.135, 0.105])
    rows = [np.array([1.0, 0, 0, 0]), phi]
    for yy, row in zip([1, 0], rows):
        left = 0
        for a, v in enumerate(row):
            if v > 0:
                ax.barh(yy, v, 0.62, left=left, color=ACT[a], ec='white', lw=0.8)
                if v > 0.3:
                    ax.text(left + v / 2, yy, f'{v:.3g}' if v < 1 else '1', ha='center', va='center', fontsize=5.4,
                            color='white')
            left += v
    ax.set_yticks([1, 0])
    ax.set_yticklabels(T['classes'], fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.grid(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=1.5)
    ax.text(1.0, -0.62, T['plan_each'].format(v=phi[1]), ha='right', va='center', fontsize=5.4, color=INK2)
    bg.text(0.5675, 0.727, T['plan_b'], ha='center', va='bottom', fontsize=5.9, color=INK2, linespacing=1.2)
    arrow(bg, (0.150, 0.8325), (0.180, 0.8325), color=INK2)
    bg.text(0.430, 0.8325, T['either'], ha='center', va='center', fontsize=7.5, weight='bold', color=INK2)
    # stage 2
    fbox(0.725, 0.718, 0.250, 0.232, fc='white', ec=INK2, lw=1.1)
    bg.text(0.85, 0.942, T['s2_t'], ha='center', va='top', fontsize=7.2, weight='bold')
    bg.text(0.85, 0.826, T['s2_b'], ha='center', va='center', fontsize=6.2, linespacing=1.35)
    arrow(bg, (0.725, 0.835), (0.680, 0.835), color=INK2, lw=1.0)

    # ---------------- fast band
    fbox(0.012, 0.010, 0.976, 0.640, fc='#f6fbf8', ec=GREEN, lw=0.8, ls=(0, (4, 3)))
    bg.text(0.025, 0.640, T['fast'], fontsize=7.4, color=GREEN, va='top', style='italic')
    bg.text(0.025, 0.616, T['example'], fontsize=6.2, color=INK2, va='top')
    # prior and counts arrows between the bands
    arrow(bg, (0.2925, 0.718), (0.2925, 0.652), color=ORANGE, lw=1.4)
    arrow(bg, (0.5675, 0.718), (0.5675, 0.652), color=BLUE, lw=1.4)
    bg.text(0.43, 0.683, T['broadcast'], fontsize=6.0, color=BLUE, va='center', ha='center', linespacing=1.2)
    arrow(bg, (0.93, 0.652), (0.93, 0.718), color=INK2, lw=1.0, ls=(0, (3, 2)))
    bg.text(0.922, 0.681, T['counts'].replace('\n', ' '), fontsize=5.8, color=INK2, va='center', ha='right')

    # step 1: mask (shared)
    fbox(0.025, 0.030, 0.150, 0.565, fc='white', ec=GREEN, lw=1.1)
    bg.text(0.10, 0.585, T['mask_t'], ha='center', va='top', fontsize=7.2, weight='bold')
    for k, (lab, why) in enumerate(zip(T['mask_rows'], T['mask_why'])):
        y = 0.525 - 0.055 * k
        ok = k < 3
        bg.add_patch(Rectangle((0.040, y - 0.013), 0.016, 0.026, fc=ACT[k], ec='none', alpha=1 if ok else 0.35))
        bg.text(0.062, y, lab, va='center', fontsize=6.4, color=INK if ok else INK2)
        bg.text(0.150, y, '✓' if ok else '✗', va='center', ha='center', fontsize=8.5,
                color=GREEN if ok else '#c0392b', weight='bold')
        if why:
            bg.text(0.062, y - 0.026, why, va='center', fontsize=5.4, color=INK2, style='italic')
    bg.text(0.10, 0.215, T['mask_b'], ha='center', va='center', fontsize=5.8, color=INK2, linespacing=1.3)

    # column headers
    cx = [0.395, 0.585, 0.775, 0.925]
    for x, t in zip(cx, T['cols']):
        bg.text(x, 0.587, t, ha='center', va='top', fontsize=6.8, weight='bold')
    # rows
    yc = [0.470, 0.280, 0.090]
    for k, (y, lab) in enumerate(zip(yc, T['rows'])):
        ec = ORANGE if k == 0 else BLUE
        fbox(0.195, y - 0.070, 0.105, 0.140, fc='white', ec=ec, lw=1.3)
        bg.text(0.2475, y, lab, ha='center', va='center', fontsize=6.4, weight='bold', color=ec, linespacing=1.25)
        arrow(bg, (0.175, y), (0.195, y), color=GREEN, lw=1.0)
        if k < 2:
            arrow(bg, (0.300, y), (0.322, y), color=INK2, lw=0.9)
            arrow(bg, (0.472, y), (0.508, y), color=INK2, lw=0.9)
            arrow(bg, (0.662, y), (0.698, y), color=INK2, lw=0.9)
        else:
            arrow(bg, (0.300, y), (0.322, y), color=INK2, lw=0.9)
            arrow(bg, (0.662, y), (0.698, y), color=INK2, lw=0.9)
        if k:
            bg.plot([0.195, 0.975], [y + 0.095, y + 0.095], color=GRID, lw=0.8)
    # row A: heuristic prior
    hh, ww = 0.115, 0.145
    mini_bars(fig, [0.325, yc[0] - 0.050, ww, hh], hA[0], note=T['a_init'])
    mini_bars(fig, [0.515, yc[0] - 0.050, ww, hh], hA[-1], before=hA[0], floor=o['lamB'] / 3, note=T['a_amp'])
    strip(fig, [0.705, yc[0] - 0.050, 0.14, hh], hA[-1], P['C2_u'][0], T)
    bg.text(0.925, yc[0], T['a_eff'], ha='center', va='center', fontsize=5.8, color=INK2, linespacing=1.25)
    # row B: planned prior
    mini_bars(fig, [0.325, yc[1] - 0.050, ww, hh], hB[0], note=T['b_init'])
    mini_bars(fig, [0.515, yc[1] - 0.050, ww, hh], hB[-1], before=hB[0], note=T['b_amp'])
    strip(fig, [0.705, yc[1] - 0.050, 0.14, hh], hB[-1], P['C2_u'][1], T)
    bg.text(0.925, yc[1], T['b_eff'], ha='center', va='center', fontsize=5.8, color=INK2, linespacing=1.25)
    # row C: per-task ranking (illustrative distribution of benefits within the class)
    ax = fig.add_axes([0.335, yc[2] - 0.050, 0.315, hh])
    x = np.linspace(-3.2, 3.2, 300)
    dens = np.exp(-x ** 2 / 2)
    from scipy.stats import norm
    tau = norm.ppf(1 - r_off)
    ax.fill_between(x, dens, color=GRID, lw=0)
    ax.fill_between(x[x >= tau], dens[x >= tau], color=BLUE, alpha=0.35, lw=0)
    ax.plot(x, dens, color=INK2, lw=0.8)
    ax.plot([tau, tau], [0, 1.3], color=BLUE, lw=1.0, ls='--')
    ax.text(tau + 0.08, 1.02, T['c_tau'], ha='left', va='bottom', fontsize=6.2, color=BLUE)
    xt = tau + 0.75
    ax.plot([xt], [0.06], marker='D', ms=4.5, color=ACT[1], mec='white', mew=0.6, zorder=5)
    ax.annotate(T['c_task'], (xt, 0.06), xytext=(xt + 0.5, 0.55), fontsize=5.6, color=INK,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    ax.text(-3.1, 2.05, T['c_rank'].format(r=r_off), ha='left', va='top', fontsize=5.6, color=INK2, linespacing=1.2)
    ax.set_ylim(0, 2.1)
    ax.set_xlim(-3.2, 3.2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(T['c_x'], fontsize=5.8, labelpad=1.5, color=INK2)
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    strip(fig, [0.705, yc[2] - 0.050, 0.14, hh], np.array([0, 0.5, 0.5]), P['C2_u'][2], T)
    bg.text(0.775, yc[2] - 0.062, T['c_pick'], ha='center', va='center', fontsize=5.4, color=INK2,
            linespacing=1.15)
    bg.text(0.925, yc[2], T['c_eff'], ha='center', va='center', fontsize=5.8, color=INK2, linespacing=1.25)
    save(fig, 'fig_c2_gipa_framework')


# ----------------------------------------------------------------------------- C3
def waffle(ax, x0, n, color, width=4, s=0.82):
    for k in range(n):
        r, c = divmod(k, width)
        ax.add_patch(Rectangle((x0 + c, r), s, s, fc=color, ec='none'))


def fig_c3():
    T = TEXT['C3']
    P = PARAMS
    N, M, xi = P['N'], P['M'], P['xi']
    nb = round((1 - P['piS']) * N)
    rng = np.random.default_rng(P['seed'])
    draw = rng.choice(4, size=nb, p=P['phi_B'])
    cnt = np.bincount(draw, minlength=4)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    W, gap = 4, 1.6
    xs = [k * (W + gap) for k in range(4)]
    cap_rows = 59 / W
    for ax, counts, ttl in [(axes[0], [0, nb, 0, 0], T['title_a']), (axes[1], list(cnt), T['title_b'])]:
        for k, n in enumerate(counts):
            waffle(ax, xs[k], n, ACT[k], width=W)
        for k in range(1, 4):
            ax.plot([xs[k] - 0.2, xs[k] + W], [cap_rows] * 2, color=INK2, ls='--', lw=0.9)
        ax.set_xlim(-0.6, xs[-1] + W + 0.4)
        ax.set_ylim(-0.5, nb / W + 2.5)
        ax.set_xticks([x + W / 2 - 0.1 for x in xs])
        ax.set_xticklabels(T['cols'])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_aspect('equal')
        ax.spines['left'].set_visible(False)
        title(ax, ttl)
        ax.annotate(T['cap'], (xs[3] + W, cap_rows), xytext=(0, -3), textcoords='offset points', ha='right',
                    va='top', fontsize=6.3, color=INK2)
    lat_a = P['eB'] + xi * (nb - 1)
    axes[0].text(xs[2] - 0.2, nb / W - 1, T['note_a'].format(lat=lat_a), fontsize=7, color=INK, va='top')
    off = int(cnt[1:].sum())
    lat_b = ((cnt[0]) * P['LB'] + sum(c * (P['eB'] + xi * (c - 1)) for c in cnt[1:]) + (N - nb) * P['LS']) / N
    axes[1].text(xs[1] - 0.2, nb / W - 1, T['note_b'].format(off=off, per=round(off / 3, 1), mx=int(cnt[1:].max()),
                                                             lat=lat_b), fontsize=7, color=INK, va='top')
    fig.text(0.01, -0.02, T['unit'].format(N=N, nb=nb), fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, 'fig_c3_herding')


# ----------------------------------------------------------------------------- C4
def fig_c4():
    T = TEXT['C4']
    P = PARAMS
    xi, eB, LB, M, N = P['xi'], P['eB'], P['LB'], P['M'], P['N']
    D = LB - eB
    nue, ns = (D + xi) / xi, (D + xi) / (2 * xi)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    n = np.linspace(0, 34, 400)
    own = eB + xi * (n - 1)
    marg = eB + xi * (2 * n - 1)
    ax.fill_between(n[n <= ns], marg[n <= ns], LB, color=BLUE, alpha=0.13, lw=0)
    ax.axhline(LB, color=INK2, ls='--', lw=1)
    ax.plot(n, own, color=ORANGE, lw=1.8)
    ax.plot(n, marg, color=BLUE, lw=1.8)
    ax.plot([nue], [LB], 'o', color=ORANGE, ms=7, mec='white', zorder=5)
    ax.plot([ns], [LB], 'o', color=BLUE, ms=7, mec='white', zorder=5)
    ax.annotate(T['local'], (0.4, LB), xytext=(0, 2), textcoords='offset points', ha='left', va='bottom', fontsize=6.8,
                color=INK2)
    ax.annotate(T['own'], (9, eB + xi * 8), xytext=(8, -5), textcoords='offset points', ha='left', va='top',
                fontsize=6.8, color=ORANGE)
    ax.annotate(T['marg'], (18.5, eB + xi * 36), xytext=(12, 0), textcoords='offset points', ha='left', va='center',
                fontsize=6.8, color=BLUE)
    ax.annotate(T['ue'].format(nue=nue), (nue, LB), xytext=(0, -30), textcoords='offset points', ha='center',
                fontsize=6.8, color=ORANGE, arrowprops=dict(arrowstyle='-', color=ORANGE, lw=0.6))
    ax.annotate(T['so'].format(ns=ns), (ns, LB), xytext=(0, 14), textcoords='offset points', ha='center',
                fontsize=6.8, color=BLUE, arrowprops=dict(arrowstyle='-', color=BLUE, lw=0.6))
    ax.annotate(T['saving'].format(sv=xi * ns ** 2), (4.0, 17.5), xytext=(0.6, 36), ha='left', va='top', fontsize=6.3,
                color=BLUE, arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
    ax.set_xlim(0, 34)
    ax.set_ylim(6, 42)
    ax.set_xlabel(T['xlabel_a'])
    ax.set_ylabel(T['ylabel_a'])
    title(ax, T['title_a'])
    ax = axes[1]
    NB, NS = (1 - P['piS']) * N, P['piS'] * N
    m = np.linspace(0, NB / M, 300)
    mean = ((NB - M * m) * LB + M * m * (eB + xi * (m - 1)) + NS * P['LS']) / N
    ax.plot(m, mean, color=INK2, lw=1.5, label=T['curve'])
    for key, col, mk in [('local', INK2, 's'), ('heur', ORANGE, 's'), ('plan', BLUE, 'o')]:
        x, y = P['measured'][key]
        ax.plot([x], [y], marker=mk, color=col, ms=7, mec='white', ls='none', label=T['pts'][key], zorder=5)
    ax.axvline(ns, color=BLUE, lw=0.8, ls=':')
    ax.axvline(nue, color=ORANGE, lw=0.8, ls=':')
    ax.set_xlabel(T['xlabel_b'])
    ax.set_ylabel(T['ylabel_b'])
    ax.set_ylim(16, 21.5)
    ax.legend(loc='upper left', fontsize=6.3)
    title(ax, T['title_b'])
    fig.tight_layout()
    save(fig, 'fig_c4_equilibrium_optimum')


# ----------------------------------------------------------------------------- C5
def operator(q, adj, p0=None, u=None, tau=0.4, beta=0.4, nu=0.3, theta0=1.5, gamma_theta=0.6, K=4, lam=0.6):
    qt = q - adj
    if p0 is None:
        e = np.exp((qt - qt.max()) / tau)
        p0 = e / e.sum()
    if u is None:
        u = np.ones_like(q) / len(q)
    p = p0.copy()
    hist = [p.copy()]
    for r in range(1, K + 1):
        th = theta0 * gamma_theta ** r
        g = np.exp(beta / (1 + np.exp(-(qt - qt.max() + th) / nu)))
        p = (1 - lam) * p * g / (p * g).sum() + lam * u
        hist.append(p.copy())
    return np.array(hist)


def fig_c5():
    T = TEXT['C5']
    P = PARAMS
    o = P['op']
    q = -np.array(P['C5_scores'])
    cbar = np.r_[0, [P['xi'] * max(P['N'] / (P['M'] + 1) - 1, 0)] * 3]
    phi = np.array(P['phi_B'])
    qbar = -np.array([P['LB'], P['eB'], P['eB'], P['eB']])
    kw = dict(tau=o['tau'], nu=o['nu'], theta0=o['theta0'], gamma_theta=o['gamma_theta'], K=o['K'])
    h_a = operator(q, o['eta'] * cbar, beta=o['beta'], lam=o['lamB'], **kw)
    h_b = operator(qbar, qbar, p0=phi, u=phi, beta=o['beta'], lam=o['lamB'], **kw)
    h_c = operator(np.array([qbar[0], qbar[1] + 1.5, qbar[2], qbar[3]]), qbar, p0=phi, u=phi,
                   beta=P['C5_beta_c'], lam=0.0, **kw)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    r = np.arange(o['K'] + 1)
    for ax, h, ttl, note, ny in [(axes[0], h_a, T['title_a'], T['note_a'], 0.85),
                                 (axes[1], h_b, T['title_b'], T['note_b'], 0.42),
                                 (axes[2], h_c, T['title_c'], T['note_c'].format(beta=P['C5_beta_c']), 0.86)]:
        for a in range(4):
            ls = '-' if a < 2 else (0, (3, 2))
            ax.plot(r, h[:, a], color=ACT[a], marker='o', ms=4, lw=1.4, ls=ls, label=T['actions'][a],
                    mec='white', mew=0.5)
        ax.set_xticks(r)
        ax.set_xlabel(T['xlabel'])
        ax.set_ylim(-0.03, 1.05)
        ax.text(0.97, ny, note, transform=ax.transAxes, ha='right', va='center', fontsize=6.3, color=INK2)
        title(ax, ttl)
    axes[0].axhline(o['lamB'] / 4, color=INK2, lw=0.8, ls=':')
    axes[0].annotate(T['floor'], (o['K'], o['lamB'] / 4), xytext=(0, -9), textcoords='offset points', ha='right',
                     fontsize=6.3, color=INK2)
    axes[0].set_ylabel(T['ylabel'])
    axes[1].legend(loc='upper right', fontsize=6.3, ncol=2, bbox_to_anchor=(1.0, 0.99))
    fig.tight_layout()
    save(fig, 'fig_c5_operator_walkthrough')


# ----------------------------------------------------------------------------- C6
def fig_c6():
    T = TEXT['C6']
    P = PARAMS
    n = P['C6_N'] - 1
    m = np.arange(0, 36)
    cap = P['C6_cap']
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    w = 0.42
    handles, labels = [], []
    for k, (key, col, off) in enumerate([('z0', ORANGE, -w / 2), ('z2', BLUE, w / 2)]):
        pmf = binom.pmf(m, n, P['C6_q'][key])
        tail = binom.sf(cap - 1, n, P['C6_q'][key])
        ok = m < cap
        ax.bar((m + off)[ok], pmf[ok], w * 0.95, color=col, alpha=0.35)      # deadline met: light
        ax.bar((m + off)[~ok], pmf[~ok], w * 0.95, color=col)                 # deadline missed: full colour
        handles.append(Rectangle((0, 0), 1, 1, fc=col))
        labels.append(T[key] + '\n' + T['tail'].format(p=tail, m=P['C6_measured'][key]))
    ax.axvline(cap - 0.5, color=INK2, ls='--', lw=1)
    ax.text(cap - 0.9, 0.158, T['cap'], ha='right', va='top', fontsize=6.5, color=INK2)
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=6.5, labelspacing=0.9)
    ax.set_xlabel(T['xlabel'])
    ax.set_ylabel(T['ylabel'])
    ax.set_xlim(-0.8, 35.8)
    ax.set_ylim(0, 0.165)
    title(ax, T['title'])
    fig.tight_layout()
    save(fig, 'fig_c6_deadline_margin')


# ----------------------------------------------------------------------------- C7
def fig_c7():
    T = TEXT['C7']
    c = np.array(PARAMS['C7_tasks'])
    F = PARAMS['C7_F']
    order = np.argsort(c)
    exact = np.array([np.minimum(c, ci).sum() / F for ci in c])
    naive = len(c) * c / F
    cols = [ACT[1], ACT[2], ACT[3], ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    t_events = np.r_[0, np.unique(exact)]
    for t0, t1 in zip(t_events[:-1], t_events[1:]):
        active = [k for k in range(len(c)) if exact[k] > t0 + 1e-9]
        y = 0
        for k in active:
            share = 1 / len(active)
            ax.add_patch(Rectangle((t0, y), t1 - t0, share, fc=cols[k], ec='white', lw=1))
            ax.text((t0 + t1) / 2, y + share / 2, f'Task {k + 1}', ha='center', va='center', fontsize=6.8,
                    color='white')                                   # direct label: identity is not colour alone
            y += share
    ax.set_xlim(0, exact.max() + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(T['xlabel_a'])
    ax.set_ylabel(T['ylabel_a'])
    ax.grid(False)
    title(ax, T['title_a'])
    ax = axes[1]
    y = np.arange(len(c))[::-1]
    ax.barh(y + 0.18, exact, 0.34, color=[cols[k] for k in range(len(c))], label=T['exact'])
    ax.barh(y - 0.18, naive, 0.34, color='white', edgecolor=[cols[k] for k in range(len(c))], hatch='////',
            lw=0.8, label=T['naive'])
    for yy, a, b in zip(y, exact, naive):
        ax.text(a + 0.4, yy + 0.18, f'{a:g}', va='center', fontsize=6.5)
        ax.text(b + 0.4, yy - 0.18, f'{b:g}', va='center', fontsize=6.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([T['task'].format(k=k + 1, c=c[k]) for k in range(len(c))], fontsize=7)
    ax.set_xlabel(T['xlabel_b'])
    ax.legend(loc='upper right', fontsize=6.5)
    title(ax, T['title_b'])
    fig.tight_layout()
    save(fig, 'fig_c7_processor_sharing')


FIGS = {1: fig_c1, 2: fig_c2, 3: fig_c3, 4: fig_c4, 5: fig_c5, 6: fig_c6, 7: fig_c7}


def main():
    ap = argparse.ArgumentParser(description='Explanatory figures')
    ap.add_argument('--figure', type=int, choices=sorted(FIGS))
    args = ap.parse_args()
    for k in ([args.figure] if args.figure else sorted(FIGS)):
        FIGS[k]()


if __name__ == '__main__':
    main()
