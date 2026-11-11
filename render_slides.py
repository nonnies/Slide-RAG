"""Render every slide of every .pptx in config.PPTX_DIR to a JPG for web.py.

  pip install pymupdf
  python render_slides.py

Needs LibreOffice (https://www.libreoffice.org). Hidden slides are rendered too, so
data/slide_images/Lecture05-12.jpg is always slide 12 of the Lecture05 deck.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import pymupdf

import config

WIDTH_PX = 1280
SOFFICE_CANDIDATES = ["soffice", r"C:\Program Files\LibreOffice\program\soffice.exe"]
EXPORT = 'pdf:impress_pdf_Export:{"ExportHiddenSlides":{"type":"boolean","value":"true"}}'


def find_soffice():
    for c in SOFFICE_CANDIDATES:
        if shutil.which(c) or Path(c).exists():
            return c
    raise SystemExit("LibreOffice not found. Install it or add soffice to SOFFICE_CANDIDATES.")


def main():
    soffice = find_soffice()
    decks = sorted(config.PPTX_DIR.glob("*.pptx"))
    if not decks:
        raise SystemExit(f"No .pptx files in {config.PPTX_DIR}. Set PPTX_DIR in config.py.")
    config.SLIDE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for deck in decks:
            subprocess.run([soffice, "--headless", "--convert-to", EXPORT, "--outdir", tmp, str(deck)],
                           check=True, capture_output=True)
            pdf = pymupdf.open(Path(tmp) / (deck.stem + ".pdf"))
            prefix = deck.stem[:9]                       # "Lecture05"
            for i, page in enumerate(pdf, 1):
                pix = page.get_pixmap(matrix=pymupdf.Matrix(WIDTH_PX / page.rect.width,
                                                         WIDTH_PX / page.rect.width))
                pix.save(str(config.SLIDE_IMAGE_DIR / f"{prefix}-{i:02d}.jpg"), jpg_quality=80)
            print(f"{deck.name}: {len(pdf)} slides")
            pdf.close()
    print(f"Images in {config.SLIDE_IMAGE_DIR}")


if __name__ == "__main__":
    main()
