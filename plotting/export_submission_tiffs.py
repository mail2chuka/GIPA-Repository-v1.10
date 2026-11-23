"""
export_submission_tiffs.py
==========================
Collects the eight figures of the revised manuscript (ITS-2026-08-0579, R1) as
one TIFF per figure, named and numbered as in the manuscript, for upload to
the journal:

    figures/submission/Figure_1.tif ... Figure_8.tif
    figures/submission/MANIFEST.txt   (size, resolution, printed width, checks)

Figs. 1, 3 and 7 and Figs. 4-6 are drawn by plotting/concept_figures.py and
plotting/revision_figures.py (600 dpi, RGB, LZW). Figs. 2 and 8 of the revised
manuscript were redrawn for the IET layout; their files are taken from the
manuscript itself, so pass the clean manuscript (.docx) as the first argument.
Fig. 8 shows the data of experiment E5b (results/res_async_local.json, the
persistent-queue experiment), the same data as figures/revision/
fig_r5b_async_local_queue.*, not the earlier fig_r5_asynchronous.*.

Checks against the Wiley figure guidelines used by IET journals
(line art 600 dpi; width 80-180 mm; under 10 MB; one figure per file named
"Figure_n"):
  * A figure wider than 180 mm at 600 dpi is re-tagged to print at 180 mm;
    the pixels are unchanged, so the effective resolution rises above 600 dpi
    and the text prints slightly smaller (the scale is reported).
  * A raster source narrower than 1800 px is flagged LOW-RES.

EDIT HERE: the FIGURES list maps manuscript figure numbers to source files;
an entry ('docx', k) takes the k-th picture embedded in the manuscript.

Run (from the repository root):
    python plotting/export_submission_tiffs.py path/to/ITS-2026-08-0579_R1_manuscript_clean.docx
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'figures' / 'revision'
CON = ROOT / 'figures' / 'concept'     # drawn by plotting/concept_figures.py (Figs. 1-3, S1, S2)
OUT = ROOT / 'figures' / 'submission'

# ======================================================================
# EDIT HERE: manuscript figure number -> source file
# ======================================================================
FIGURES = [
    (1, CON / 'fig_c1_system_architecture.tif', 'System model'),
    (2, ('docx', 2), 'GIPA stages and the three decision rules'),
    (3, CON / 'fig_c4_equilibrium_optimum.tif', 'User equilibrium and system optimum'),
    (4, SRC / 'fig_r1_main_comparison.tif', 'Mean latency against fleet size, both latency models'),
    (5, SRC / 'fig_r2_prediction_error.tif', 'Robustness to prediction error'),
    (6, SRC / 'fig_r3_capacity.tif', 'RSU load and admission'),
    (7, SRC / 'fig_r6_learning_automata.tif', 'Learning automata'),
    (8, ('docx', 8), 'Asynchronous arrivals with persistent queues (E5b)'),
]
PDF_WIDTH_MM = 170.0      # PDF sources are rasterised to this printed width at TARGET_DPI
TARGET_DPI = 600          # line art
MAX_WIDTH_MM = 180.0
MIN_WIDTH_PX = 1800
MAX_MB = 10.0
# ======================================================================

MM_PER_IN = 25.4


def rasterise_pdf(pdf, width_mm):
    """First page of a PDF at TARGET_DPI for the given printed width, whitespace cropped."""
    import subprocess
    import tempfile
    from PIL import ImageChops
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(['pdftoppm', '-png', '-r', '800', '-f', '1', '-l', '1', '-singlefile', str(pdf), f'{d}/p'], check=True)
        im = Image.open(f'{d}/p.png').convert('RGB')
    bbox = ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()
    im = im.crop((max(bbox[0] - 24, 0), max(bbox[1] - 24, 0), min(bbox[2] + 24, im.width), min(bbox[3] + 24, im.height)))
    w = round(width_mm / MM_PER_IN * TARGET_DPI)
    return im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)


def docx_picture(path, k):
    """The k-th picture embedded in a .docx, in document order."""
    import io
    import zipfile
    import docx
    from docx.oxml.ns import qn
    d = docx.Document(path)
    rels = {r.rId: r.target_ref for r in d.part.rels.values()}
    names = [rels[b.get(qn('r:embed'))] for b in d.element.body.iter(qn('a:blip'))]
    return Image.open(io.BytesIO(zipfile.ZipFile(path).read('word/' + names[k - 1]))).convert('RGB')


def main():
    import sys
    manuscript = sys.argv[1] if len(sys.argv) > 1 else None
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ['Manuscript ITS-2026-08-0579, revision R1: figure files for submission (eight figures)', '',
             f'{"File":<17}{"Pixels":>13}{"dpi":>6}{"Width mm":>10}{"MB":>7}  Status  Content']
    problems = []
    for n, src, what in FIGURES:
        if isinstance(src, tuple):
            if manuscript is None:
                raise SystemExit(f'Figure_{n} is taken from the manuscript: pass the clean .docx as the first argument')
            im = docx_picture(manuscript, src[1])
            src = Path(manuscript)
        elif not src.exists():
            raise SystemExit(f'missing {src}: run plotting/revision_figures.py and plotting/gipa_framework.py first')
        elif src.suffix.lower() == '.pdf':
            im = rasterise_pdf(src, PDF_WIDTH_MM)
        else:
            im = Image.open(src).convert('RGB')
        w, h = im.size
        status = 'ok'
        if src.suffix.lower() == '.png' and w < MIN_WIDTH_PX:
            # raster original below the minimum size: keep its pixels, tag its native resolution
            dpi = float(im.info.get('dpi', (96, 96))[0]) if src.suffix.lower() == '.png' else TARGET_DPI
            dpi = round(dpi) or 96
            status = 'LOW-RES'
            problems.append(f'Figure_{n}: {w} x {h} px only (minimum {MIN_WIDTH_PX} px wide). '
                            'Supply the original drawing or a redraw.')
        else:
            dpi = TARGET_DPI
            width_mm = w / dpi * MM_PER_IN
            if width_mm > MAX_WIDTH_MM:
                dpi = int(-(-w * MM_PER_IN // MAX_WIDTH_MM))  # ceiling: prints at <= 180 mm
                status = f'x{TARGET_DPI / dpi:.2f}'
                if TARGET_DPI / dpi < 0.95:  # report only a noticeable reduction
                    problems.append(f'Figure_{n}: {width_mm:.0f} mm wide at {TARGET_DPI} dpi; tagged {dpi} dpi to print '
                                    f'at {MAX_WIDTH_MM:.0f} mm, so text prints at {TARGET_DPI / dpi:.0%} of its set size.')
        dst = OUT / (f'Figure_{n}.tif' if isinstance(n, int) else f'{n}.tif')
        im.save(dst, format='TIFF', compression='tiff_lzw', dpi=(dpi, dpi))
        mb = dst.stat().st_size / 1e6
        if mb > MAX_MB:
            problems.append(f'Figure_{n}: {mb:.1f} MB exceeds {MAX_MB:.0f} MB')
        lines.append(f'{dst.name:<17}{f"{w}x{h}":>13}{dpi:>6}{w / dpi * MM_PER_IN:>10.0f}{mb:>7.2f}  {status:<7} {what}')
        print(lines[-1])
    lines += ['', 'All files: RGB, LZW-compressed TIFF, white background.', '']
    lines += ['Checks:'] + ([f'  - {p}' for p in problems] if problems else ['  none'])
    (OUT / 'MANIFEST.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[-len(problems) - 1:]))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
