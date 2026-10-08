"""Verify a rendered PDF still contains the text of the notebook it came from.

WeasyPrint does not always clip overflowing content visibly -- it can drop the
overflowing part of a line outright. That failure leaves nothing to see on the
page and nothing to measure with a bounding-box check, because deleted glyphs
have no bounding box. A page can look perfect and be missing words mid-sentence.

The only reliable check is to compare the text actually extracted from the PDF
against the text that went in. This takes consecutive word runs from the source
notebook as probes and reports what fraction survives into the PDF, listing the
ones that did not so the damage is identifiable rather than merely quantified.

Usage: marjum_pdf_verify.py NOTEBOOK.ipynb RENDERED.pdf [--probe-words 6] [--show 25]
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def normalise(text):
    """Collapse to comparable form: PDF line breaks and spacing differ from source."""
    text = unicodedata.normalize('NFKC', text)
    # Typographic substitutions the renderer may introduce.
    for a, b in (('’', "'"), ('‘', "'"), ('“', '"'), ('”', '"'),
                 ('—', '-'), ('–', '-'), ('−', '-'), (' ', ' ')):
        text = text.replace(a, b)
    # Whitespace is REMOVED, not collapsed. A soft wrap inside a long token
    # comes back from text extraction as an inserted space, so collapsing to a
    # single space makes every wrap point look like altered text. Removing
    # whitespace entirely makes wrapping invisible while still detecting any
    # glyph that WeasyPrint actually dropped.
    return re.sub(r'\s+', '', text).lower()


def source_text(nb_path):
    """Text expected to appear in the PDF: markdown prose, code, and text outputs."""
    nb = json.loads(Path(nb_path).read_text())
    chunks = []
    for cell in nb['cells']:
        src = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            # Markdown is rendered, so drop the syntax that will not survive.
            src = re.sub(r'`{1,3}', '', src)
            src = re.sub(r'\*{1,3}', '', src)
            src = re.sub(r'^#{1,6}\s*', '', src, flags=re.M)
            src = re.sub(r'^\s*[-|>]\s*', '', src, flags=re.M)
            # Ordered-list markers become <ol> list markers, which the browser
            # box model draws and PDF text extraction does not emit. Keeping
            # the literal "1." in the expected text makes every numbered list
            # look like deleted prose.
            src = re.sub(r'^\s*\d+\.\s+', '', src, flags=re.M)
            # A table delimiter row ("|---|---:|") draws the header rule and
            # emits no text at all. Left in, its dashes look like a deletion.
            src = re.sub(r'^\s*\|?[\s\-:|]*\|[\s\-:|]*$\n?', '', src, flags=re.M)
            src = src.replace('|', ' ')
            chunks.append(src)
        else:
            chunks.append(src)
            for out in cell.get('outputs', []):
                if 'text' in out:
                    chunks.append(''.join(out['text']))
                data = out.get('data', {})
                # A text/plain alongside an image is the object repr
                # ("<Figure size ...>"), which nbconvert deliberately does not
                # render. Counting it as expected text would be a false alarm.
                if 'text/plain' in data and 'image/png' not in data:
                    chunks.append(''.join(data['text/plain']))
    return chunks


def probes(chunks, n_words=6):
    """Consecutive word runs, deduplicated, long enough to be unambiguous."""
    seen, out = set(), []
    for chunk in chunks:
        words = re.sub(r'\s+', ' ', chunk).split()
        for i in range(0, max(0, len(words) - n_words + 1)):
            p = normalise(' '.join(words[i:i + n_words]))
            if len(p) < 20 or p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def verify(nb_path, pdf_path, n_words=6, show=25, stride=1):
    import fitz
    doc = fitz.open(str(pdf_path))
    pages = [normalise(page.get_text()) for page in doc]
    pdf_text = normalise(' '.join(page.get_text() for page in doc))
    doc.close()

    all_probes = probes(source_text(nb_path), n_words)
    probe_list = all_probes[::stride]
    absent = [p for p in probe_list if p not in pdf_text]

    # A probe that straddles a page break cannot match the concatenated text:
    # extraction puts the rest of the page (often another cell's output)
    # between the two halves. That is fragmentation, not deletion.
    #
    # The same happens WITHIN a page: extraction emits a markdown table, then a
    # code cell, then that cell's output, so a probe running from the last table
    # cell into the following paragraph is also split by intervening content.
    #
    # The halves are therefore constrained by ADJACENCY rather than by length:
    # the first half on page i, the second on page i or i+1. An earlier version
    # instead required each half to be at least 4 characters, which is both too
    # weak (a long half can still match by luck) and too strong -- a break
    # falling three characters into a word ("...remedy" / "for separated
    # modes") was reported as deleted prose.
    fragmented, missing = [], []
    for p in absent:
        split_ok = any(p[:k] in pages[i] and p[k:] in pages[i + j]
                       for k in range(2, len(p) - 1)
                       for i in range(len(pages))
                       for j in (0, 1) if i + j < len(pages))
        (fragmented if split_ok else missing).append(p)

    kept = len(probe_list) - len(missing)
    pct = 100.0 * kept / len(probe_list) if probe_list else 100.0

    print(f'{Path(pdf_path).name}')
    print(f'  probes: {len(probe_list)} of {len(all_probes)} ({n_words}-word runs from the notebook)')
    print(f'  intact: {kept}  ({len(fragmented)} split across a page break, both halves present)')
    print(f'  genuinely missing: {len(missing)}  -> {pct:.2f}% text intact')
    if missing:
        print(f'  first {min(show, len(missing))} missing probe(s):')
        for p in missing[:show]:
            print(f'    - {p!r}')
    print(f'  VERDICT: {"PASS - no text lost" if not missing else "FAIL - text missing from the PDF"}')
    return pct, missing


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('notebook')
    ap.add_argument('pdf')
    ap.add_argument('--probe-words', type=int, default=6)
    ap.add_argument('--show', type=int, default=25)
    ap.add_argument('--stride', type=int, default=1,
                    help='sample every Nth probe; 1 checks all of them')
    a = ap.parse_args()
    pct, missing = verify(a.notebook, a.pdf, a.probe_words, a.show, a.stride)
    raise SystemExit(0 if not missing else 1)
