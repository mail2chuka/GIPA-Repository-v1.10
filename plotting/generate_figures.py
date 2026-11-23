#!/usr/bin/env python3
"""Generate Figs. 3-8 with spacious labels and archived 95% bootstrap CIs.

generate_figures.py
===================
Draws manuscript Figs. 3-8 from the archived statistical summaries. It does
not run the simulator: every bar, point and error bar comes from
``results/res_main_statistics.json`` (key ``summaries``), which holds, for
each regime, density, policy and metric, the across-seed mean and the
seed-level 95% percentile-bootstrap interval (10,000 resamples).

Figures (each saved as PNG at 300 dpi, PDF and SVG)
---------------------------------------------------
Fig. 3  fig3_matched_policy_comparison_N160   Regime A, N = 160: miss rates,
        latency, offload ratio and Jain fairness for all 10 policies.
Fig. 4  fig4_class_specific_offload_N160      Regime A, N = 160: safety and
        background offload ratios.
Fig. 5  fig5_gipa_density_all_metrics         GIPA across N = 80-320.
Fig. 6  fig6_gipa_ablation_all_metrics        GIPA ablations, Regime A, N = 160.
Fig. 7  fig7_energy_and_peak_load_N160        Energy and maximum RSU load.
Fig. 8  fig8_regimeB_diagnostic_N160          Regime B, N = 160 diagnostic.

Usage
-----
    python plotting/generate_figures.py                 # all figures
    python plotting/generate_figures.py --figure 7      # one figure
    python plotting/generate_figures.py --statistics PATH --output DIR

Defaults: statistics = results/res_main_statistics.json, output = figures/.

Metric keys (see gipa_core.run): mS, mB miss rates; lat, p95 latency (ms);
off, offS, offB offload ratios; fair Jain index; en stylized energy; maxl
maximum RSU load. Rates are multiplied by 100 for display where noted.
A metric whose mean is ``null`` (e.g. Jain fairness when nothing is
offloaded) is drawn as missing.
"""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

# Policy keys in plotting order, their display labels and bar colours.
P = ['greedy', 'indep', 'rbr', 'price', 'msoft', 'msoft_prior', 'entreg', 'ql', 'local', 'gipa']
L = ['Greedy', 'Independent softmax', 'Reactive best response', 'Congestion price + rounding',
     'Masked softmax', 'Prior-adjusted masked softmax', 'Entropy-regularized masked',
     'Tabular Q-learning', 'All-local', 'GIPA']
C = ['#707070', '#C44E52', '#8172B3', '#8C6D5A', '#DD8452', '#B7A62A', '#4C9F9A',
     '#4C72B0', '#222222', '#18864B']
plt.rcParams.update({'font.size': 9.5, 'axes.titlesize': 11, 'axes.labelsize': 10,
                     'legend.fontsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})


def z(S, r, n, p, m, scale=1):
    """Return (mean, asymmetric error bar) for one summary entry.

    Parameters: S summaries dict, r regime ('A'/'B'), n density, p policy key,
    m metric key, scale display multiplier (100 for percentages).
    The error bar is a (2, 1) array [[mean - lo], [hi - mean]] for
    matplotlib's ``xerr``/``yerr``. ``None`` values become NaN.
    """
    q = S[r][str(n)][p][m]
    mean = np.nan if q['mean'] is None else q['mean'] * scale
    lo = np.nan if q['ci95'][0] is None else q['ci95'][0] * scale
    hi = np.nan if q['ci95'][1] is None else q['ci95'][1] * scale
    return mean, np.array([[mean - lo], [hi - mean]])


def vec(S, r, n, ps, m, scale=1):
    """Vectorised :func:`z` over a list of policies: (means, (2, k) error array)."""
    a = [z(S, r, n, p, m, scale) for p in ps]
    means = np.array([x[0] for x in a])
    err = np.hstack([x[1] for x in a])
    return means, err


def clean(a):
    """Light x-grid, grid behind bars, no top or right spines."""
    a.grid(axis='x', alpha=.22)
    a.set_axisbelow(True)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)


def save(f, o, name):
    """Save figure ``f`` as o/name.{png,pdf,svg} (PNG at 300 dpi) and close it."""
    for e in ('png', 'pdf', 'svg'):
        f.savefig(o / f'{name}.{e}', dpi=300 if e == 'png' else None,
                  bbox_inches='tight', facecolor='white')
    plt.close(f)


def f3(S, o):
    """Fig. 3: matched policy comparison, Regime A, N = 160 (four panels)."""
    y = np.arange(len(P)); h = .36
    f, a = plt.subplots(2, 2, figsize=(14.8, 10.2))
    f.subplots_adjust(left=.25, right=.98, top=.91, bottom=.08, hspace=.28, wspace=.22)
    # (a) deadline-miss rate per class
    ms, es = vec(S, 'A', 160, P, 'mS', 100)
    mb, eb = vec(S, 'A', 160, P, 'mB', 100)
    a[0, 0].barh(y - h / 2, ms, h, xerr=es, label='Safety')
    a[0, 0].barh(y + h / 2, mb, h, xerr=eb, label='Background')
    a[0, 0].set(title='(a) Deadline-miss rate', xlabel='Miss rate (%)')
    a[0, 0].legend()
    # (b) mean and p95 latency
    la, el = vec(S, 'A', 160, P, 'lat')
    p95, ep = vec(S, 'A', 160, P, 'p95')
    a[0, 1].barh(y - h / 2, la, h, xerr=el, label='Mean')
    a[0, 1].barh(y + h / 2, p95, h, xerr=ep, label='p95')
    a[0, 1].set(title='(b) Latency', xlabel='Latency (ms)')
    a[0, 1].legend()
    # (c) overall offload ratio
    off, eo = vec(S, 'A', 160, P, 'off', 100)
    a[1, 0].barh(y, off, xerr=eo, color=C)
    a[1, 0].set(title='(c) Overall offload ratio', xlabel='Offloaded tasks (%)')
    # (d) Jain fairness; policies with undefined fairness are omitted and marked n/a
    fa, ef = vec(S, 'A', 160, P, 'fair')
    m = np.isfinite(fa)
    a[1, 1].barh(y[m], fa[m], xerr=ef[:, m], color=np.array(C)[m])
    a[1, 1].set(title='(d) Jain fairness over RSU loads', xlabel='Jain index', xlim=(0, 1.05))
    a[1, 1].text(.03, 8, 'n/a', va='center')
    for r in range(2):
        a[r, 0].set_yticks(y, L); a[r, 0].invert_yaxis()
        a[r, 1].set_yticks(y, []); a[r, 1].invert_yaxis()
    for x in a.ravel():
        clean(x)
    f.suptitle('Regime A at $N=160$: means and seed-level 95% bootstrap CIs', fontsize=15)
    save(f, o, 'fig3_matched_policy_comparison_N160')


def f4(S, o):
    """Fig. 4: safety and background offload ratios, Regime A, N = 160."""
    y = np.arange(len(P)); h = .36
    f, a = plt.subplots(figsize=(12, 7.2))
    f.subplots_adjust(left=.28, right=.97, top=.89, bottom=.11)
    s, es = vec(S, 'A', 160, P, 'offS', 100)
    b, eb = vec(S, 'A', 160, P, 'offB', 100)
    a.barh(y - h / 2, s, h, xerr=es, label='Safety tasks')
    a.barh(y + h / 2, b, h, xerr=eb, label='Background tasks')
    a.set_yticks(y, L); a.invert_yaxis()
    a.set(xlabel='Offloaded tasks (%)', title='Class-specific offload ratios in Regime A at $N=160$',
          xlim=(0, 106))
    a.legend(); clean(a)
    save(f, o, 'fig4_class_specific_offload_N160')


def f5(S, o):
    """Fig. 5: GIPA across fleet density N = 80-320, Regime A (six panels)."""
    ns = [80, 160, 240, 320]
    f, a = plt.subplots(2, 3, figsize=(15.2, 8.5))
    f.subplots_adjust(left=.07, right=.98, top=.89, bottom=.10, hspace=.42, wspace=.30)

    def ln(ax, m, label=None, scale=1, marker='o'):
        """Plot GIPA's metric ``m`` against N with CI error bars on axis ``ax``."""
        q = [z(S, 'A', n, 'gipa', m, scale) for n in ns]
        v = np.array([x[0] for x in q])
        e = np.hstack([x[1] for x in q])
        ax.errorbar(ns, v, yerr=e, capsize=3, marker=marker, label=label)

    ln(a[0, 0], 'lat', 'Mean'); ln(a[0, 0], 'p95', 'p95', marker='s')
    a[0, 0].set(title='(a) Latency', ylabel='Latency (ms)'); a[0, 0].legend()
    ln(a[0, 1], 'off', 'Overall', 100); ln(a[0, 1], 'offS', 'Safety', 100, 's')
    ln(a[0, 1], 'offB', 'Background', 100, '^')
    a[0, 1].set(title='(b) Offload ratio', ylabel='Offloaded tasks (%)'); a[0, 1].legend()
    ln(a[0, 2], 'fair')
    a[0, 2].set(title='(c) Jain fairness', ylabel='Jain index')
    a[0, 2].text(280, .915, 'n/a at 240 and 320', ha='center')
    ln(a[1, 0], 'en'); a[1, 0].set(title='(d) Stylized energy', ylabel='Energy units per task')
    ln(a[1, 1], 'maxl'); a[1, 1].set(title='(e) Maximum RSU load', ylabel='Tasks')
    ln(a[1, 2], 'mS', 'Safety', 100); ln(a[1, 2], 'mB', 'Background', 100, 's')
    a[1, 2].set(title='(f) Deadline-miss rate', ylabel='Miss rate (%)'); a[1, 2].legend()
    for x in a.ravel():
        x.set_xlabel('Fleet size, $N$'); x.set_xticks(ns); x.grid(alpha=.22)
    f.suptitle('GIPA across fleet density: means and seed-level 95% bootstrap CIs', fontsize=15)
    save(f, o, 'fig5_gipa_density_all_metrics')


def f6(S, o):
    """Fig. 6: GIPA component ablation, Regime A, N = 160 (four panels)."""
    ps = ['gipa', 'gipa_noamp', 'gipa_nodiff', 'gipa_fixed', 'gipa_binary', 'gipa_noprior']
    ll = ['Full GIPA', 'Without amplification', 'Without diffusion', 'Fixed weights',
          'Binary weights', 'Without prior']
    y = np.arange(6); h = .36
    f, a = plt.subplots(2, 2, figsize=(13.8, 8.5))
    f.subplots_adjust(left=.25, right=.98, top=.90, bottom=.09, hspace=.30, wspace=.24)
    # (a) latency and (b) offload: two bars per variant
    for ax, m1, m2, t, xlab, scale in [(a[0, 0], 'lat', 'p95', '(a) Latency', 'Latency (ms)', 1),
                                       (a[0, 1], 'off', 'offB', '(b) Offload ratio',
                                        'Offloaded tasks (%)', 100)]:
        v, e = vec(S, 'A', 160, ps, m1, scale)
        w, ew = vec(S, 'A', 160, ps, m2, scale)
        ax.barh(y - h / 2, v, h, xerr=e, label='Mean' if m1 == 'lat' else 'Overall')
        ax.barh(y + h / 2, w, h, xerr=ew, label='p95' if m2 == 'p95' else 'Background')
        ax.set(title=t, xlabel=xlab)
        ax.legend()
    # (c) energy and (d) peak load: one bar per variant
    for ax, m, t, xlab in [(a[1, 0], 'en', '(c) Stylized energy', 'Energy units per task'),
                           (a[1, 1], 'maxl', '(d) Maximum RSU load', 'Tasks')]:
        v, e = vec(S, 'A', 160, ps, m)
        ax.barh(y, v, xerr=e)
        ax.set(title=t, xlabel=xlab)
    for r in range(2):
        a[r, 0].set_yticks(y, ll); a[r, 0].invert_yaxis()
        a[r, 1].set_yticks(y, []); a[r, 1].invert_yaxis()
    for x in a.ravel():
        clean(x)
    f.suptitle('GIPA component ablation at $N=160$: means and 95% bootstrap CIs', fontsize=15)
    save(f, o, 'fig6_gipa_ablation_all_metrics')


def f7(S, o):
    """Fig. 7: stylized energy and maximum RSU load, Regime A, N = 160."""
    y = np.arange(len(P))
    f, a = plt.subplots(1, 2, figsize=(14.5, 7))
    f.subplots_adjust(left=.25, right=.98, top=.87, bottom=.10, wspace=.24)
    for ax, m, t, xlab in [(a[0], 'en', '(a) Stylized energy', 'Energy units per task'),
                           (a[1], 'maxl', '(b) Maximum RSU load', 'Tasks')]:
        v, e = vec(S, 'A', 160, P, m)
        ax.barh(y, v, xerr=e, color=C)
        ax.set(title=t, xlabel=xlab)
        ax.invert_yaxis()
        clean(ax)
    a[0].set_yticks(y, L); a[1].set_yticks(y, [])
    f.suptitle('Energy and peak-load diagnostics in Regime A at $N=160$', fontsize=15)
    save(f, o, 'fig7_energy_and_peak_load_N160')


def f8(S, o):
    """Fig. 8: Regime B diagnostic at N = 160 (miss rate, latency, class offload)."""
    ps = ['local', 'ql', 'msoft', 'price', 'rbr', 'gipa']
    ll = ['All-local', 'Tabular Q-learning', 'Masked softmax', 'Congestion price + rounding',
          'Reactive best response', 'GIPA']
    y = np.arange(6); h = .36
    f, a = plt.subplots(1, 3, figsize=(16, 6.2))
    f.subplots_adjust(left=.23, right=.98, top=.86, bottom=.12, wspace=.25)
    s, es = vec(S, 'B', 160, ps, 'mS', 100)
    b, eb = vec(S, 'B', 160, ps, 'mB', 100)
    a[0].barh(y - h / 2, s, h, xerr=es, label='Safety')
    a[0].barh(y + h / 2, b, h, xerr=eb, label='Background')
    a[0].set(title='(a) Deadline-miss rate', xlabel='Miss rate (%)'); a[0].legend()
    v, e = vec(S, 'B', 160, ps, 'lat')
    a[1].barh(y, v, xerr=e)
    a[1].set(title='(b) Mean latency', xlabel='Latency (ms)')
    s, es = vec(S, 'B', 160, ps, 'offS', 100)
    b, eb = vec(S, 'B', 160, ps, 'offB', 100)
    a[2].barh(y - h / 2, s, h, xerr=es, label='Safety')
    a[2].barh(y + h / 2, b, h, xerr=eb, label='Background')
    a[2].set(title='(c) Class-specific offload', xlabel='Offloaded tasks (%)'); a[2].legend()
    for i, x in enumerate(a):
        x.set_yticks(y, ll if i == 0 else []); x.invert_yaxis(); clean(x)
    f.suptitle('Regime B diagnostic at $N=160$: means and 95% bootstrap CIs', fontsize=15)
    save(f, o, 'fig8_regimeB_diagnostic_N160')


# Figure number -> drawing function.
F = {3: f3, 4: f4, 5: f5, 6: f6, 7: f7, 8: f8}


def main():
    """Parse arguments, load the archived summaries and draw the requested figures."""
    p = argparse.ArgumentParser()
    p.add_argument('--figure', type=int, choices=F)
    p.add_argument('--statistics', type=Path)
    p.add_argument('--output', type=Path)
    q = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    src = q.statistics or root / 'results/res_main_statistics.json'
    o = q.output or root / 'figures'
    o.mkdir(exist_ok=True)
    S = json.load(open(src))['summaries']
    nums = [q.figure] if q.figure else sorted(F)
    for n in nums:
        F[n](S, o)
    print('Generated', len(nums), 'figure set(s) in PNG, PDF, and SVG.')


if __name__ == '__main__':
    main()
