#!/usr/bin/env python3
"""Check that no text was dropped between the rendered HTML and the PDF.

WeasyPrint DELETES text that overflows the page box rather than clipping it,
so words vanish from mid-sentence and the PDF still looks structurally fine.
Every render in this directory must run this afterwards. Usage:

    python verify_pdf_text.py <basename>     # expects <basename>.{html,pdf}
"""
import re
import sys
from html.parser import HTMLParser

try:
    from pypdf import PdfReader
except ImportError:                                    # older env
    from PyPDF2 import PdfReader


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buf, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if self.skip == 0:
            self.buf.append(data)


def main(base):
    parser = _Text()
    with open(f"{base}.html") as f:
        parser.feed(f.read())
    html_txt = " ".join("".join(parser.buf).split())
    pdf_txt = " ".join(" ".join(
        (page.extract_text() or "") for page in PdfReader(f"{base}.pdf").pages
    ).split())

    # A word containing a hyphen can be broken across lines AT that hyphen, so
    # the extracted PDF text reads "self- consistent" where the HTML says
    # "self-consistent". That is a line break, not lost text, and it produced a
    # false positive that cost real time to chase. Allow optional whitespace
    # after any hyphen when matching.
    words = set(re.findall(r"[A-Za-z][A-Za-z\-]{4,}", html_txt))

    def present(w):
        if w in pdf_txt:
            return True
        if "-" not in w:
            return False
        return re.search(re.escape(w).replace(r"\-", r"-\s*"), pdf_txt) is not None

    missing = sorted(w for w in words if not present(w))
    print(f"{base}: {len(words)} distinct words in HTML, "
          f"{len(missing)} missing from PDF")
    if missing:
        print("  missing:", missing[:40])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "rfi_flag_prototype"))
