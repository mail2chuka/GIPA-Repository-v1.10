"""
gipa_framework.py
=================
Draws the block diagram of the GIPA framework: Stage 0 (heuristic or planned
prior), Stage 1 (per-vehicle probability operator: mask, initialise,
amplify and diffuse, sample) and Stage 2 (prior refresh). Manuscript Fig. 2
of the first revision of ITS-2026-08-0579.

Output: figures/revision/fig_gipa_framework.{png,pdf,svg,tif}
(PNG 300 dpi; TIFF 600 dpi, RGB, LZW-compressed)

WHERE TO EDIT THE WORDING
-------------------------
Every word in the diagram is in the TEXT block marked "EDIT HERE" below:
stage titles and subtitles, the text of each box, and the arrow labels.
Use "\\n" for a line break; text between $...$ is typeset as mathematics.
Box sizes are fixed, so a much longer text needs a shorter wording or an
extra line break rather than more width.

Run (from the repository root):
    python plotting/gipa_framework.py
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures' / 'revision' / 'fig_gipa_framework'

INK = '#1f2328'
INK2 = '#57606a'
BLUE = '#2a78d6'
ORANGE = '#e8702a'
GREEN = '#1aa37a'
FILL = {'s0': '#e8f1fc', 's1': '#eaf7f2', 's2': '#fdf0e7'}
EDGE = {'s0': BLUE, 's1': GREEN, 's2': ORANGE}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})

# ======================================================================
# EDIT HERE: wording of the diagram
# ======================================================================
TEXT = {
    # Stage 0 (left column)
    's0_title': 'Stage 0: prior',
    's0_sub': 'offline or at each refresh',
    's0_inputs': 'Inputs: anticipated fleet size $\\bar N$,\nclass mix $\\pi_k$, latency model',
    's0_heuristic': 'Heuristic prior: delay if vehicles\nspread evenly,\n$\\bar c_j=\\xi_j\\max\\{\\bar N/(M+1)-1,0\\}$',
    's0_or': 'or',
    's0_planned': 'Planned prior: solve P1 for the\nsystem-optimal plan $\\phi_k$ under\nthe margin $z\\sigma_j$; class scores $\\bar q_k$',
    # Stage 1 (middle column)
    's1_title': 'Stage 1: probability operator',
    's1_sub': 'every vehicle, every task; no messages',
    's1_mask': 'Feasibility mask $\\mathcal{F}_i(t)$ from predicted\nlatencies $\\hat\\ell_{i,a}$, deadline and coverage',
    's1_init': 'Initialise $p^{(0)}$: softmax of prior-adjusted\nscores, or the plan $\\phi_k$ restricted to $\\mathcal{F}_i$',
    's1_amp': '$K$ rounds: amplify near-best actions,\nthen diffuse towards the target $u_i$',
    's1_sample': 'Sample $a_i \\sim p^{(K)}$: local or RSU $j$',
    # Stage 2 (right column)
    's2_title': 'Stage 2: prior refresh',
    's2_sub': 'slow timescale, every $T_p$ slots',
    's2_counts': 'RSU association counts\n(aggregate, per class)',
    's2_update': 'Update $\\bar N$ and class mix;\nrecompute the prior\nif they change',
    's2_broadcast': 'Broadcast the prior\n(planned: 60 B/s;\nheuristic: 90 B/s)',
    # arrow labels
    'arrow_prior': 'prior',
    'arrow_refresh': 'refreshed prior',
}
FORMATS = ['png', 'pdf', 'svg', 'tif']
PNG_DPI = 300
TIFF_DPI = 600
# ======================================================================
# end of EDIT HERE
# ======================================================================


def panel(ax, x, y, w, h, key, title, subtitle):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                fc=FILL[key], ec=EDGE[key], lw=1.4))
    ax.text(x + w / 2, y + h - 0.14, title, ha='center', va='top', fontsize=8.6, weight='bold', color=INK)
    ax.text(x + w / 2, y + h - 0.42, subtitle, ha='center', va='top', fontsize=7.0, color=INK2, style='italic')


def box(ax, x, y, w, h, text, ec=INK2, fs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.04',
                                fc='white', ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.35)


def arrow(ax, p, q, color=INK2, ls='-', rad=0.0, lw=1.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}'))


def main():
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(-0.2, 5.0)
    ax.axis('off')
    W0, W1, W2 = 3.6, 4.2, 3.6
    x0, x1, x2 = 0.1, 0.1 + W0 + 0.3, 0.1 + W0 + 0.3 + W1 + 0.3
    bw0, bw1, bw2 = W0 - 0.4, W1 - 0.4, W2 - 0.4

    panel(ax, x0, 0.45, W0, 4.45, 's0', TEXT['s0_title'], TEXT['s0_sub'])
    box(ax, x0 + 0.2, 3.35, bw0, 0.75, TEXT['s0_inputs'], EDGE['s0'])
    box(ax, x0 + 0.2, 2.05, bw0, 0.95, TEXT['s0_heuristic'], EDGE['s0'])
    box(ax, x0 + 0.2, 0.7, bw0, 1.0, TEXT['s0_planned'], EDGE['s0'])
    arrow(ax, (x0 + W0 / 2, 3.35), (x0 + W0 / 2, 3.0))
    ax.text(x0 + W0 / 2, 1.85, TEXT['s0_or'], ha='center', va='center', fontsize=7.2, color=INK2, style='italic')

    panel(ax, x1, 0.45, W1, 4.45, 's1', TEXT['s1_title'], TEXT['s1_sub'])
    box(ax, x1 + 0.2, 3.35, bw1, 0.75, TEXT['s1_mask'], EDGE['s1'])
    box(ax, x1 + 0.2, 2.35, bw1, 0.65, TEXT['s1_init'], EDGE['s1'])
    box(ax, x1 + 0.2, 1.35, bw1, 0.65, TEXT['s1_amp'], EDGE['s1'])
    box(ax, x1 + 0.2, 0.65, bw1, 0.42, TEXT['s1_sample'], EDGE['s1'])
    for ya, yb in [(3.35, 3.0), (2.35, 2.0), (1.35, 1.07)]:
        arrow(ax, (x1 + W1 / 2, ya), (x1 + W1 / 2, yb))

    panel(ax, x2, 0.45, W2, 4.45, 's2', TEXT['s2_title'], TEXT['s2_sub'])
    box(ax, x2 + 0.2, 3.35, bw2, 0.75, TEXT['s2_counts'], EDGE['s2'])
    box(ax, x2 + 0.2, 2.05, bw2, 0.95, TEXT['s2_update'], EDGE['s2'])
    box(ax, x2 + 0.2, 0.7, bw2, 1.0, TEXT['s2_broadcast'], EDGE['s2'])
    arrow(ax, (x2 + W2 / 2, 3.35), (x2 + W2 / 2, 3.0))
    arrow(ax, (x2 + W2 / 2, 2.05), (x2 + W2 / 2, 1.7))

    # plan from Stage 0 to Stage 1
    arrow(ax, (x0 + 0.2 + bw0, 2.5), (x1 + 0.2, 2.65), color=BLUE, lw=1.5)
    arrow(ax, (x0 + 0.2 + bw0, 1.2), (x1 + 0.2, 2.45), color=BLUE, lw=1.5)
    ax.text((x0 + W0 + x1) / 2 - 0.05, 2.85, TEXT['arrow_prior'], ha='center', fontsize=7.2, color=BLUE)
    # offloaded tasks feed association counts (elbow above the panels)
    ax.plot([x1 + 0.2 + bw1, x1 + W1 + 0.15, x1 + W1 + 0.15], [0.86, 0.86, 3.72], color=ORANGE, lw=1.1, ls='--')
    arrow(ax, (x1 + W1 + 0.15, 3.72), (x2 + 0.2, 3.72), color=ORANGE, ls='--')
    # refreshed plan back to Stage 1 (route below the panels)
    ax.plot([x2 + W2 / 2, x2 + W2 / 2, x1 + W1 * 0.8], [0.7, 0.2, 0.2], color=ORANGE, lw=1.1)
    arrow(ax, (x1 + W1 * 0.8, 0.2), (x1 + W1 * 0.8, 0.65), color=ORANGE)
    ax.text(x2 - 0.1, 0.02, TEXT['arrow_refresh'], ha='center', fontsize=7.2, color=ORANGE)
    for ext in FORMATS:
        if ext == 'tif':
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=TIFF_DPI, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            Image.open(buf).convert('RGB').save(OUT.with_suffix('.tif'), format='TIFF', compression='tiff_lzw',
                                                dpi=(TIFF_DPI, TIFF_DPI))
        else:
            fig.savefig(OUT.with_suffix('.' + ext), dpi=PNG_DPI, bbox_inches='tight')
    print('saved', OUT)


if __name__ == '__main__':
    main()
