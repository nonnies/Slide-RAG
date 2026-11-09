"""
Slide -> RAG test pipeline (first N slides)

1. python-pptx reads NATIVE content exactly (no OCR):
     text, tables, charts, equations, speaker notes
2. Pictures in the slides are extracted and sent to the PaddleOCR-VL pipeline.
   Its layout detection labels each region (chart / table / formula / text)
   and uses the right prompt automatically, via your running llama-server.
3. One Markdown file per slide is written, ready for RAG chunking.

Run this in a SEPARATE conda env (not your PyTorch env):
    conda create -n paddleocr python=3.11 -y
    conda activate paddleocr
    python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    python -m pip install -U "paddleocr[doc-parser]>=3.6.0"
    python -m pip install python-pptx pillow

Start llama-server (port 8090) first, then:
    python slide_rag_test.py
"""

import io
import time
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

# ---------------- CONFIG ----------------
PPTX_PATH = r"pptx\Lecture01-Introduction (HEU-NLP).pptx"    # <-- change this
OUTPUT_DIR = Path("slide_rag_output")
MAX_SLIDES = 10                              # test on first 10 slides
SERVER_URL = "http://127.0.0.1:8090/v1"      # your llama-server
DEVICE = "gpu:0"                             # layout model device; use "cpu" if VRAM is tight
MIN_WIDTH_CM = 3.0                           # skip small pictures (icons, logos)
MAX_SIDE_PX = 2000                           # downscale huge images for speed
# ----------------------------------------

EMU_PER_CM = 360_000
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


# ---------- native extraction (python-pptx) ----------

def iter_shapes(shapes):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from iter_shapes(sh.shapes)
        else:
            yield sh


def table_to_md(table):
    rows = [[cell.text.replace("\n", " ").strip() for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    md = ["| " + " | ".join(rows[0]) + " |",
          "| " + " | ".join("---" for _ in rows[0]) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(md)


def chart_to_md(chart):
    lines = []
    try:
        if chart.has_title and chart.chart_title.text_frame.text:
            lines.append(f"Chart title: {chart.chart_title.text_frame.text}")
    except Exception:
        pass
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


def equations_in_slide(slide):
    """Native Office Math equations, as linear text (not full LaTeX)."""
    eqs = []
    for om in slide._element.iter(f"{{{MATH_NS}}}oMath"):
        txt = "".join(t.text or "" for t in om.iter(f"{{{MATH_NS}}}t")).strip()
        if txt and txt not in eqs:
            eqs.append(txt)
    return eqs


def get_picture(sh):
    try:
        return sh.image
    except (AttributeError, ValueError, KeyError):
        return None


def to_png(blob, path):
    img = Image.open(io.BytesIO(blob))
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > MAX_SIDE_PX:
        img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    img.save(path, format="PNG")
    return img.size


def extract_native(slide):
    title, texts, tables, charts, pictures = "", [], [], [], []
    for sh in iter_shapes(slide.shapes):
        if getattr(sh, "has_chart", False) and sh.has_chart:
            charts.append(chart_to_md(sh.chart))
        elif getattr(sh, "has_table", False) and sh.has_table:
            tables.append(table_to_md(sh.table))
        elif getattr(sh, "has_text_frame", False) and sh.has_text_frame and sh.text_frame.text.strip():
            t = sh.text_frame.text.strip()
            if sh.is_placeholder and "TITLE" in str(sh.placeholder_format.type) and not title:
                title = t
            else:
                texts.append(t)
        else:
            img = get_picture(sh)
            if img is not None and sh.width and sh.width >= MIN_WIDTH_CM * EMU_PER_CM:
                pictures.append(img)
    notes = ""
    if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
        notes = slide.notes_slide.notes_text_frame.text.strip()
    return {"title": title, "texts": texts, "tables": tables, "charts": charts,
            "equations": equations_in_slide(slide), "pictures": pictures, "notes": notes}


# ---------- PaddleOCR-VL pipeline ----------

def build_pipeline():
    from paddleocr import PaddleOCRVL
    kwargs = dict(pipeline_version="v1.6",
                  vl_rec_backend="llama-cpp-server",
                  vl_rec_server_url=SERVER_URL,
                  device=DEVICE)
    try:
        # chart recognition is off by default in some versions; turn it on
        return PaddleOCRVL(use_chart_recognition=True, **kwargs)
    except TypeError:
        return PaddleOCRVL(**kwargs)


def ocr_picture(pipeline, img_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for res in pipeline.predict(str(img_path)):
        res.save_to_markdown(save_path=str(out_dir))
    parts = [p.read_text(encoding="utf-8") for p in sorted(out_dir.glob("*.md"))]
    return "\n".join(parts).strip()


# ---------- main ----------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    img_dir = OUTPUT_DIR / "images"
    img_dir.mkdir(exist_ok=True)

    prs = Presentation(PPTX_PATH)
    total_slides = len(prs.slides)
    n_slides = min(MAX_SLIDES, total_slides)

    print("Loading PaddleOCR-VL pipeline (layout model + llama-server backend)...")
    t_load = time.perf_counter()
    pipeline = build_pipeline()
    print(f"Pipeline ready in {time.perf_counter() - t_load:.1f}s\n")

    for res in pipeline.predict(str(img_dir)):
        res.save_to_markdown(save_path=str(OUTPUT_DIR / "detected_boxes"))
        res.save_to_img(save_path=str(OUTPUT_DIR/ "detected_boxes"))   # saves image with layout boxes and labels

    ocr_times, t_start = [], time.perf_counter()
    stats = {"tables": 0, "charts": 0, "equations": 0, "pictures": 0, "failed": 0}

    for n, slide in enumerate(prs.slides, 1):
        if n > n_slides:
            break
        d = extract_native(slide)
        stats["tables"] += len(d["tables"])
        stats["charts"] += len(d["charts"])
        stats["equations"] += len(d["equations"])

        md = [f"# Slide {n}" + (f": {d['title']}" if d["title"] else ""), ""]
        md += [f"<!-- source: {Path(PPTX_PATH).name}, slide {n} -->", ""]
        if d["texts"]:
            md += ["## Text", *d["texts"], ""]
        for i, t in enumerate(d["tables"], 1):
            md += [f"## Table {i} (native)", t, ""]
        for i, c in enumerate(d["charts"], 1):
            md += [f"## Chart {i} (native data)", c, ""]
        if d["equations"]:
            md += ["## Equations (native)", *[f"- {e}" for e in d["equations"]], ""]

        for i, img in enumerate(d["pictures"], 1):
            img_path = img_dir / f"slide_{n:03d}_pic_{i:02d}.png"
            try:
                size = to_png(img.blob, img_path)
            except Exception as e:
                print(f"[slide {n}] picture {i} skipped ({img.ext}): {e}")
                stats["failed"] += 1
                continue

            t0 = time.perf_counter()
            try:
                text = ocr_picture(pipeline, img_path,
                                   OUTPUT_DIR / "ocr_raw" / f"slide_{n:03d}_pic_{i:02d}")
            except Exception as e:
                text = f"OCR failed: {e}"
                stats["failed"] += 1
            dt = time.perf_counter() - t0
            ocr_times.append(dt)
            stats["pictures"] += 1
            print(f"[slide {n}] picture {i} ({size[0]}x{size[1]}) OCR'd in {dt:.1f}s")
            md += [f"## Picture {i} (OCR: {img_path.name})", text or "(no content detected)", ""]

        if d["notes"]:
            md += ["## Speaker notes", d["notes"], ""]

        (OUTPUT_DIR / f"slide_{n:03d}.md").write_text("\n".join(md), encoding="utf-8")
        print(f"[slide {n}] saved slide_{n:03d}.md")

    elapsed = time.perf_counter() - t_start
    print("\n========== SUMMARY ==========")
    print(f"Slides processed  : {n_slides} of {total_slides}")
    print(f"Native tables     : {stats['tables']}")
    print(f"Native charts     : {stats['charts']}")
    print(f"Native equations  : {stats['equations']}")
    print(f"Pictures OCR'd    : {stats['pictures']}  (failed/skipped: {stats['failed']})")
    print(f"Total time        : {elapsed:.1f}s")
    if ocr_times:
        avg = sum(ocr_times) / len(ocr_times)
        all_pics = sum(
            1 for s in prs.slides for sh in iter_shapes(s.shapes)
            if not (getattr(sh, "has_chart", False) and sh.has_chart)
            and get_picture(sh) is not None and sh.width and sh.width >= MIN_WIDTH_CM * EMU_PER_CM
        )
        print(f"Avg per picture   : {avg:.1f}s")
        print(f"Full deck estimate: {all_pics} pictures x {avg:.1f}s = ~{all_pics * avg / 60:.1f} min")
    print(f"Output folder     : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()