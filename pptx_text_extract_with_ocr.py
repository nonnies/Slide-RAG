"""
Test pipeline: extract charts from the first N slides of a .pptx

- Native PowerPoint charts  -> data read directly from the file (no OCR)
- Chart pictures            -> sent to llama-server (PaddleOCR-VL) with "Chart Recognition:"

Requirements (safe in your PyTorch env):
    pip install python-pptx requests pillow

Start llama-server first (port 8090), then run:
    python chart_ocr_test.py
"""

import base64
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

# ---------------- CONFIG ----------------
PPTX_PATH = r"pptx\Lecture01-Introduction (HEU-NLP).pptx"   # <-- change this
OUTPUT_DIR = Path("raw_ocr_output")
MAX_SLIDES = 10                             # test on first 10 slides
SERVER_URL = "http://127.0.0.1:8090/v1/chat/completions"
PROMPT = "OCR:"
MIN_WIDTH_CM = 5.5                          # skip small pictures (icons, logos)
MAX_SIDE_PX = 2000                          # downscale huge images for speed
REQUEST_TIMEOUT = 300                       # seconds per image
# ----------------------------------------

EMU_PER_CM = 360_000


def iter_shapes(shapes):
    """Yield all shapes, including those inside groups."""
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from iter_shapes(sh.shapes)
        else:
            yield sh


def get_picture(sh):
    """Return the image object if the shape holds a picture, else None."""
    try:
        return sh.image  # works for pictures and picture placeholders
    except (AttributeError, ValueError, KeyError):
        return None


def to_png_bytes(blob):
    """Convert any image to PNG (handles EMF/WMF/TIFF/BMP where Pillow can) and downscale."""
    img = Image.open(io.BytesIO(blob))
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > MAX_SIDE_PX:
        img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.size


def ocr_chart(png_bytes):
    b64 = base64.b64encode(png_bytes).decode()
    r = requests.post(SERVER_URL, timeout=REQUEST_TIMEOUT, json={
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "temperature": 0,
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def read_native_chart(chart):
    """Read exact data from a native PowerPoint chart."""
    lines = []
    try:
        title = chart.chart_title.text_frame.text if chart.has_title else ""
    except Exception:
        title = ""
    if title:
        lines.append(f"Title: {title}")
    lines.append(f"Type: {chart.chart_type}")
    for plot in chart.plots:
        try:
            cats = [str(c) for c in plot.categories]
        except Exception:
            cats = []
        for s in plot.series:
            vals = list(s.values)
            if cats and len(cats) == len(vals):
                pairs = ", ".join(f"{c}={v}" for c, v in zip(cats, vals))
            else:
                pairs = ", ".join(str(v) for v in vals)
            lines.append(f"- {s.name}: {pairs}")
    return "\n".join(lines)


def is_candidate(sh):
    return get_picture(sh) is not None and sh.width and sh.width >= MIN_WIDTH_CM * EMU_PER_CM


def main():
    # Check server is up
    try:
        requests.get(SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=5)
    except requests.RequestException:
        print("Cannot reach llama-server on port 8090. Start it first.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    prs = Presentation(PPTX_PATH)
    total_slides = len(prs.slides)

    # Count candidate pictures in the whole deck (for the full-run estimate)
    total_candidates = sum(
        1 for slide in prs.slides for sh in iter_shapes(slide.shapes)
        if not getattr(sh, "has_chart", False) and is_candidate(sh)
    )

    results, ocr_times = [], []
    t_start = time.perf_counter()

    for n, slide in enumerate(prs.slides, 1):
        if n > MAX_SLIDES:
            break
        slide_items = []

        for idx, sh in enumerate(iter_shapes(slide.shapes), 1):
            # 1) Native chart: no OCR needed
            if getattr(sh, "has_chart", False) and sh.has_chart:
                slide_items.append({"slide": n, "kind": "native_chart",
                                    "text": read_native_chart(sh.chart)})
                print(f"[slide {n}] native chart read directly")
                continue

            # 2) Picture: OCR if large enough
            img = get_picture(sh)
            if img is None:
                continue
            if not is_candidate(sh):
                continue

            try:
                png, size = to_png_bytes(img.blob)
            except Exception as e:
                print(f"[slide {n}] skipped picture ({img.ext}): cannot convert ({e})")
                slide_items.append({"slide": n, "kind": "skipped",
                                    "text": f"Unsupported image format: {img.ext}"})
                continue

            # save the image so you can compare it with the OCR output
            img_path = OUTPUT_DIR / f"slide_{n:03d}_img_{idx:02d}.png"
            img_path.write_bytes(png)

            t0 = time.perf_counter()
            try:
                text = ocr_chart(png)
            except Exception as e:
                text = f"OCR failed: {e}"
            dt = time.perf_counter() - t0
            ocr_times.append(dt)

            print(f"[slide {n}] OCR {size[0]}x{size[1]} in {dt:.1f}s")
            slide_items.append({"slide": n, "kind": "ocr_chart", "image": img_path.name,
                                "seconds": round(dt, 2), "text": text})

        # One Markdown file per slide (handy for RAG chunks later)
        if slide_items:
            md = [f"# Slide {n}\n"]
            for it in slide_items:
                md.append(f"## {it['kind']}" + (f" ({it['image']})" if "image" in it else ""))
                md.append(it["text"] + "\n")
            (OUTPUT_DIR / f"slide_{n:03d}.md").write_text("\n".join(md), encoding="utf-8")
        results.extend(slide_items)

    elapsed = time.perf_counter() - t_start
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Summary ----------------
    print("\n========== SUMMARY ==========")
    print(f"Slides processed : {min(MAX_SLIDES, total_slides)} of {total_slides}")
    print(f"Native charts    : {sum(r['kind'] == 'native_chart' for r in results)}")
    print(f"Pictures OCR'd   : {len(ocr_times)}")
    print(f"Skipped pictures : {sum(r['kind'] == 'skipped' for r in results)}")
    print(f"Total time       : {elapsed:.1f}s")
    if ocr_times:
        avg = sum(ocr_times) / len(ocr_times)
        print(f"Avg per picture  : {avg:.1f}s")
        est = avg * total_candidates
        print(f"Full deck estimate: {total_candidates} pictures x {avg:.1f}s "
              f"= ~{est / 60:.1f} min")
    print(f"Output folder    : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()