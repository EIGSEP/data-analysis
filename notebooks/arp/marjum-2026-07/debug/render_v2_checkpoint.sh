#!/bin/bash
# Render the executed checkpoint notebook to PDF.
#
# The width CSS below is not cosmetic. WeasyPrint DELETES text that overflows
# the page box rather than clipping it, so words vanish from mid-sentence and
# bounding-box checks still report clean. It needs an explicit LENGTH width
# (percentages never bind, because the containing block is itself
# unconstrained) plus min-width:0 on the flex cell wrappers. min-width:0 alone
# does not fix it. Always run the text-integrity verify afterwards.
set -euo pipefail
cd "$(dirname "$0")"

MAMBA=/home/aparsons/.local/share/mamba/envs/arp
export LD_LIBRARY_PATH=$MAMBA/lib
PY=$MAMBA/bin/python3
NB=beam_fits_v2_review_checkpoint

$PY -m jupyter nbconvert --to html --template lab --no-input "$NB.ipynb"

cat > /tmp/nb_pdf_fix.css <<'CSS'
@page { size: A4 landscape; margin: 1.5cm; }
html, body, main, .jp-Notebook {
  width: 820px !important; max-width: 820px !important;
  margin: 0 !important; padding: 0 !important; overflow: visible !important;
}
.jp-Cell-inputWrapper, .jp-Cell-outputWrapper, .jp-InputArea,
.jp-OutputArea, .jp-OutputArea-child, .jp-OutputArea-output,
.jp-RenderedHTMLCommon, .jp-Cell, .jp-Notebook > * {
  min-width: 0 !important; max-width: 820px !important;
  overflow: visible !important;
}
.jp-RenderedHTMLCommon table { width: auto !important; font-size: 8.5pt; }
pre, code { white-space: pre-wrap !important; word-break: break-word !important;
            font-size: 8pt !important; }
img { max-width: 800px !important; height: auto !important; }
.jp-Cell { page-break-inside: avoid; }
h1, h2 { page-break-after: avoid; }
/* Keep each list item whole. Without this a list that straddles a page break
   leaves its "1." / "2." markers stranded at the foot of the previous page
   while the item text starts the next one -- the content is all there, but the
   numbering reads as detached. */
li { break-inside: avoid; page-break-inside: avoid; }
ol, ul { break-inside: auto; page-break-inside: auto; }
CSS

$PY - "$NB" <<'PYEOF'
import sys
from weasyprint import HTML, CSS
nb = sys.argv[1]
HTML(f"{nb}.html").write_pdf(f"{nb}.pdf",
                             stylesheets=[CSS("/tmp/nb_pdf_fix.css")])
print(f"wrote {nb}.pdf")
PYEOF

ls -la "$NB.pdf"
file "$NB.pdf"
$PY /tmp/strong_verify.py "$NB.html" "$NB.pdf"
