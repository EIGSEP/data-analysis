"""Render a notebook HTML export to PDF without clipping, then verify it.

The default weasyprint render of an nbconvert HTML export clips content at the
right margin: notebook code cells are `white-space: pre` inside a horizontally
scrollable box, which in print has nowhere to scroll, and figures wider than the
content box are cut rather than scaled. Both were shipping silently because
`ls -la`/`file`/page-count all pass on a clipped PDF.

This injects print CSS that wraps preformatted text and bounds image width, then
renders a sample of pages to PNG so the result can actually be looked at.

Looking at pages is NOT sufficient verification. WeasyPrint can drop the
overflowing part of a line outright rather than clip it visibly, which leaves
nothing to see and no bounding box to measure. Run marjum_pdf_verify.py against
the rendered PDF, and run it BEFORE the shrink step: shrinking rasterises every
page, which removes all extractable text and makes text verification impossible
(and the shipped file unsearchable).

Usage: marjum_pdf_render.py IN.html OUT.pdf [--target-mb 3] [--check-pages 1 3 5]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# Orientation is per-document, not global. Landscape suits wide figures (the
# per-camera pages are 11.5x8.6in), but hurts prose or tall figures: a tall
# near-square figure will not fit a landscape page height and gets pushed to
# the next page, inflating page count and leaving large gaps.
PRINT_CSS = """
<style>
@page { size: A4 __ORIENTATION__; margin: 1.5cm; }
/* The nbconvert lab template never constrains content to the print page box.
   A percentage max-width is useless here because it resolves against an
   already-unconstrained containing block, so the fix needs an explicit length.
   __CONTENT_PX__ is computed from this page size and margin -- recompute it if
   either changes rather than copying a number from another document. */
html, body, main, .jp-Notebook {
    width: __CONTENT_PX__px !important;
    max-width: __CONTENT_PX__px !important;
    overflow-x: hidden !important;
    /* Without border-box the template's own padding is added OUTSIDE the width
       above and pushes content back past the page edge -- which measurably made
       clipping worse than no width at all. */
    box-sizing: border-box !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}
/* Everything inside inherits border-box so nested padding cannot re-overflow. */
.jp-Notebook, .jp-Notebook * { box-sizing: border-box !important; }
/* Markdown/prose containers need the explicit length too. A percentage here
   resolves against a parent the template leaves unconstrained, which is exactly
   the trap: measured, prose still ran ~38 pt past the margin with only
   percentage caps in place, while code and output were already fine. */
.jp-MarkdownOutput, .jp-RenderedMarkdown, .jp-RenderedHTMLCommon,
.jp-Cell, .jp-Cell-inputWrapper, .jp-Cell-outputWrapper, .jp-InputArea,
.jp-OutputArea, .jp-OutputArea-child, .jp-OutputArea-output, .jp-Notebook-cell {
    max-width: __CONTENT_PX__px !important;
    overflow-x: hidden !important;
}
/* The "In [n]:" / "Out[n]:" gutter is ~55pt of fixed width per cell that the
   flex row adds beside the content, pushing the right edge past the page
   margin. It carries no information in a review document, so drop it in print
   and give the width back to the content. */
.jp-InputPrompt, .jp-OutputPrompt { display: none !important; width: 0 !important; }
/* Flex/grid children default to min-width:auto, which lets them refuse to
   shrink below their content and blow back through the width above. */
.jp-Cell-inputWrapper, .jp-Cell-outputWrapper, .jp-InputArea, .jp-OutputArea,
.jp-OutputArea-child, .jp-InputArea-editor, .jp-RenderedHTMLCommon,
.jp-OutputArea-output, .jp-Cell {
    min-width: 0 !important;
}
html, body { overflow: visible !important; }
/* Preformatted code and output must wrap: in print there is no scrollbar, so
   anything that overflows is simply lost off the right edge. */
pre, code, .highlight, .highlight pre, .jp-RenderedText pre,
.jp-OutputArea-output pre, .CodeMirror-line, .jp-InputArea-editor {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-wrap: break-word !important;
    overflow: visible !important;
    max-width: 100% !important;
}
/* nbconvert wraps cells in scroll containers; make them print-visible. */
.jp-Cell, .jp-InputArea, .jp-OutputArea, .jp-OutputArea-child,
.jp-InputArea-editor, .jp-Cell-inputWrapper, .jp-Cell-outputWrapper,
.jp-RenderedHTMLCommon, div.output_subarea {
    overflow: visible !important;
    max-width: 100% !important;
}
/* Scale figures down to the content box instead of cropping them. */
img, svg, canvas { max-width: 100% !important; height: auto !important; }
/* NOT `table-layout: fixed`. Fixed layout divides the width equally regardless
   of content, and WeasyPrint then DELETES the content of a cell that does not
   fit its share rather than wrapping it -- measured: a whole 4-column table
   cell ("azimuth offset, horizontal position") vanished from the rendered page
   with no visible gap. Auto layout sizes columns to content; the width cap
   below still stops the table running off the page. */
table { table-layout: auto !important; width: auto !important; max-width: 100% !important; }
td, th {
    word-wrap: break-word !important;
    overflow-wrap: anywhere !important;
    white-space: normal !important;   /* cells must be allowed to wrap */
}
/* Slightly tighter monospace so wide fixed-width tables fit before wrapping. */
pre, code { font-size: 8.4pt !important; line-height: 1.25 !important; }
</style>
"""


# A4 in cm; CSS px are 1/96 inch.
A4_CM = (21.0, 29.7)
MARGIN_CM = 1.5


def content_px(orientation='portrait', margin_cm=MARGIN_CM):
    """Printable width in CSS px for this page size and margin."""
    short, long_ = A4_CM
    width_cm = long_ if orientation == 'landscape' else short
    return int(round((width_cm - 2 * margin_cm) / 2.54 * 96))


def inject(html_path, out_html=None, orientation='portrait'):
    html = Path(html_path).read_text(errors='replace')
    out_html = Path(out_html or str(html_path).replace('.html', '.print.html'))
    css = (PRINT_CSS.replace('__ORIENTATION__', orientation)
                    .replace('__CONTENT_PX__', str(content_px(orientation))))
    if '</head>' in html:
        html = html.replace('</head>', css + '</head>', 1)
    else:
        html = css + html
    out_html.write_text(html)
    return out_html


def render(html_path, pdf_path, target_mb=3.0, check_pages=(0, 2), orientation='portrait'):
    # weasyprint and PyMuPDF cannot be imported into the same interpreter here:
    # PyMuPDF pulls in a libLerc built against a newer libstdc++ than the system
    # one, and PIL's _imaging then fails to load. Shell weasyprint out instead.
    import subprocess
    import sys

    import fitz
    printable = inject(html_path, orientation=orientation)
    subprocess.run([sys.executable, '-m', 'weasyprint', str(printable), str(pdf_path)],
                   check=True, capture_output=True)
    size = Path(pdf_path).stat().st_size / 1e6
    if size > target_mb:
        from marjum_pdf_shrink import shrink
        tmp = Path(str(pdf_path) + '.full')
        Path(pdf_path).rename(tmp)
        shrink(tmp, pdf_path, target_mb=target_mb)
        tmp.unlink()
        size = Path(pdf_path).stat().st_size / 1e6
    printable.unlink(missing_ok=True)

    doc = fitz.open(str(pdf_path))
    pngs = []
    for i in check_pages:
        if i < len(doc):
            p = f'{Path(pdf_path).stem}_check_p{i + 1}.png'
            doc[i].get_pixmap(dpi=100).save(p)
            pngs.append(p)
    print(f'{pdf_path}: {size:.2f} MB, {len(doc)} pages; wrote {len(pngs)} check PNG(s): {pngs}')
    doc.close()
    return pngs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('html')
    ap.add_argument('pdf')
    ap.add_argument('--target-mb', type=float, default=3.0)
    ap.add_argument('--check-pages', type=int, nargs='*', default=[0, 2])
    ap.add_argument('--orientation', choices=['portrait', 'landscape'], default='portrait',
                    help='landscape for wide-figure documents, portrait for prose/tall figures')
    a = ap.parse_args()
    render(a.html, a.pdf, a.target_mb, a.check_pages, a.orientation)
