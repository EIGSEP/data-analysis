"""Rasterize a PDF's pages to keep review artifacts under a size budget.

A notebook full of trace plots and scatter maps exports to vector PDF at tens of
megabytes, which is impractical to send for review. Rendering each page to a
JPEG at moderate quality and rebuilding the PDF keeps the visual content while
collapsing the size; text stops being selectable, which is an acceptable trade
for a figure-dominated review artifact (the .ipynb and .html keep the live text).

Usage: marjum_pdf_shrink.py IN.pdf OUT.pdf [--target-mb 3] [--dpi 110] [--quality 75]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def shrink(src, dst, target_mb=3.0, dpi=110, quality=75, min_dpi=55):
    src, dst = Path(src), Path(dst)
    start_mb = src.stat().st_size / 1e6
    while True:
        doc = fitz.open(src)
        out = fitz.open()
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = pix.tobytes('jpeg', jpg_quality=quality)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=img)
        out.save(dst, deflate=True, garbage=4)
        out.close()
        doc.close()
        size_mb = dst.stat().st_size / 1e6
        print(f'{src.name}: {start_mb:.1f} MB -> {size_mb:.1f} MB at {dpi} dpi q{quality}')
        if size_mb <= target_mb or dpi <= min_dpi:
            return size_mb
        # Step down resolution first, then quality; resolution buys more.
        dpi = max(min_dpi, int(dpi * 0.8))
        quality = max(50, quality - 5)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--target-mb', type=float, default=3.0)
    ap.add_argument('--dpi', type=int, default=110)
    ap.add_argument('--quality', type=int, default=75)
    a = ap.parse_args()
    shrink(a.src, a.dst, a.target_mb, a.dpi, a.quality)
