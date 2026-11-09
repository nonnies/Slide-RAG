"""
Slide -> RAG test pipeline (first N slides)

1. python-pptx reads NATIVE content exactly (no OCR):
     text, tables, charts, equations, speaker notes
2. Pictures in the slides go through the PaddleOCR-VL pipeline.
   Layout detection labels each region (text / table / formula / chart / image)
   and the right prompt is used automatically via your llama-server.
3. NEW: regions labeled "image" (diagrams, speech bubbles, illustrations) are
   cropped and sent to llama-server with "OCR:", so their text is no longer lost.
4. One Markdown file per slide is written, ready for RAG chunking.

Env (separate from PyTorch):
    python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    python -m pip install -U "paddleocr[doc-parser]>=3.6.0"
    python -m pip install python-pptx pillow requests

Start llama-server (port 8090) first, then:
    python slide_rag_test.py
"""

# --- SSL fix: drop CA-bundle env vars that point to missing files (must run before other imports)
import os
import certifi
for _k in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    _v = os.environ.get(_k)
    if _v is not None and not os.path.exists(_v):
        os.environ.pop(_k)
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

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
PPTX_PATH = r"pptx\Lecture01-Introduction (HEU-NLP).pptx"    # <-- change this
OUTPUT_DIR = Path("slide_rag_output")
MAX_SLIDES = 10                              # test on first 10 slides
SERVER_BASE = "http://127.0.0.1:8090/v1"     # your llama-server
DEVICE = "gpu:0"                             # layout model device; "cpu" if VRAM is tight
MIN_WIDTH_CM = 3.0                           # skip small pictures (icons, logos)
MAX_SIDE_PX = 2000                           # downscale huge images for speed
IMAGE_LABELS = {"image", "figure"}           # region labels that get "OCR:" fallback
MIN_REGION_PX = 40                           # skip tiny image regions
CROP_PAD_PX = 8                              # padding around cropped regions
SAVE_LAYOUT_VIS = True                       # save images with layout boxes (for checking)
REQUEST_TIMEOUT = 300
# ----------------------------------------

EMU_PER_CM = 360_000
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
CHAT_URL = SERVER_BASE + "/chat/completions"


# ================= native extraction (python-pptx) =================

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


def is_candidate_picture(sh):
    return get_picture(sh) is not None and sh.width and sh.width >= MIN_WIDTH_CM * EMU_PER_CM


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
        elif is_candidate_picture(sh):
            pictures.append(get_picture(sh))
    notes = ""
    if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
        notes = slide.notes_slide.notes_text_frame.text.strip()
    return {"title": title, "texts": texts, "tables": tables, "charts": charts,
            "equations": equations_in_slide(slide), "pictures": pictures, "notes": notes}


# ================= OCR =================

def direct_ocr(pil_img, prompt="OCR:"):
    """Send one image straight to llama-server with a fixed PaddleOCR-VL prompt."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    r = requests.post(CHAT_URL, timeout=REQUEST_TIMEOUT, json={
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def build_pipeline():
    from paddleocr import PaddleOCRVL
    kwargs = dict(pipeline_version="v1.6",
                  vl_rec_backend="llama-cpp-server",
                  vl_rec_server_url=SERVER_BASE,
                  device=DEVICE)
    try:
        return PaddleOCRVL(use_chart_recognition=True, **kwargs)
    except TypeError:
        return PaddleOCRVL(**kwargs)


def _res_dict(res):
    j = res.json() if callable(getattr(res, "json", None)) else getattr(res, "json", {})
    return j.get("res", j) if isinstance(j, dict) else {}


def get_blocks(res):
    """Return the pipeline's regions in reading order: [{label, bbox, content}]."""
    r = _res_dict(res)
    blocks = []
    for b in r.get("parsing_res_list") or []:
        if isinstance(b, dict):
            label = b.get("block_label") or b.get("label") or ""
            bbox = b.get("block_bbox") or b.get("bbox")
            content = b.get("block_content") or b.get("content") or ""
        else:
            label = getattr(b, "label", "") or getattr(b, "block_label", "")
            bbox = getattr(b, "bbox", None) or getattr(b, "block_bbox", None)
            content = getattr(b, "content", "") or getattr(b, "block_content", "")
        blocks.append({"label": str(label).lower(), "bbox": bbox, "content": (content or "").strip()})
    return blocks


def format_block(label, content):
    if not content:
        return ""
    if "title" in label:
        return f"**{content}**"
    if "formula" in label and not content.startswith("$"):
        return f"$$ {content} $$"
    return content


def ocr_picture(pipeline, img_path, out_dir):
    """Layout pipeline + OCR: fallback on image regions. Returns (markdown, stats)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    stats = {"regions": 0, "image_regions_ocrd": 0, "labels": []}
    parts = []

    results = list(pipeline.predict(str(img_path)))
    for res in results:
        if SAVE_LAYOUT_VIS:
            try:
                res.save_to_img(save_path=str(out_dir))
            except Exception:
                pass
        try:
            res.save_to_json(save_path=str(out_dir))
        except Exception:
            pass

        blocks = get_blocks(res)
        stats["regions"] += len(blocks)
        stats["labels"] += [b["label"] for b in blocks]
        known_text = "\n".join(b["content"] for b in blocks if b["label"] not in IMAGE_LABELS)

        for b in blocks:
            if b["label"] in IMAGE_LABELS and b["bbox"]:
                x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"][:4]]
                x1, y1 = max(0, x1 - CROP_PAD_PX), max(0, y1 - CROP_PAD_PX)
                x2, y2 = min(W, x2 + CROP_PAD_PX), min(H, y2 + CROP_PAD_PX)
                if x2 - x1 < MIN_REGION_PX or y2 - y1 < MIN_REGION_PX:
                    continue
                crop = pil.crop((x1, y1, x2, y2))
                try:
                    text = direct_ocr(crop, "OCR:")
                except Exception as e:
                    text = f"(OCR failed on image region: {e})"
                # drop lines already captured by the text regions (avoids duplicates)
                lines = [ln for ln in text.splitlines() if ln.strip() and ln.strip() not in known_text]
                if lines:
                    stats["image_regions_ocrd"] += 1
                    parts.append("[text inside diagram/image]\n" + "\n".join(lines))
            else:
                formatted = format_block(b["label"], b["content"])
                if formatted:
                    parts.append(formatted)

    # Safety net: layout found nothing usable -> OCR the whole picture
    if not "".join(parts).strip():
        try:
            parts = ["[whole-picture OCR]\n" + direct_ocr(pil, "OCR:")]
        except Exception as e:
            parts = [f"OCR failed: {e}"]

    return "\n\n".join(parts).strip(), stats


# ================= main =================

def main():
    try:
        requests.get(SERVER_BASE.replace("/v1", "") + "/health", timeout=5)
    except requests.RequestException:
        print("Cannot reach llama-server on port 8090. Start it first.")
        return

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

    ocr_times, t_start = [], time.perf_counter()
    stats = {"tables": 0, "charts": 0, "equations": 0, "pictures": 0,
             "image_regions_ocrd": 0, "failed": 0}
    log = []

    for n, slide in enumerate(prs.slides, 1):
        if n > n_slides:
            break
        d = extract_native(slide)
        stats["tables"] += len(d["tables"])
        stats["charts"] += len(d["charts"])
        stats["equations"] += len(d["equations"])

        md = [f"# Slide {n}" + (f": {d['title']}" if d["title"] else ""), "",
              f"<!-- source: {Path(PPTX_PATH).name}, slide {n} -->", ""]
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
                text, s = ocr_picture(pipeline, img_path,
                                      OUTPUT_DIR / "ocr_raw" / f"slide_{n:03d}_pic_{i:02d}")
            except Exception as e:
                text, s = f"OCR failed: {e}", {"regions": 0, "image_regions_ocrd": 0, "labels": []}
                stats["failed"] += 1
            dt = time.perf_counter() - t0
            ocr_times.append(dt)
            stats["pictures"] += 1
            stats["image_regions_ocrd"] += s["image_regions_ocrd"]
            log.append({"slide": n, "picture": img_path.name, "seconds": round(dt, 2),
                        "labels": s["labels"], "image_regions_ocrd": s["image_regions_ocrd"]})
            print(f"[slide {n}] picture {i} ({size[0]}x{size[1]}) {dt:.1f}s | "
                  f"regions: {s['labels']} | image regions OCR'd: {s['image_regions_ocrd']}")
            md += [f"## Picture {i} (OCR: {img_path.name})", text or "(no content detected)", ""]

        if d["notes"]:
            md += ["## Speaker notes", d["notes"], ""]

        (OUTPUT_DIR / f"slide_{n:03d}.md").write_text("\n".join(md), encoding="utf-8")
        print(f"[slide {n}] saved slide_{n:03d}.md")

    (OUTPUT_DIR / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    print("\n========== SUMMARY ==========")
    print(f"Slides processed    : {n_slides} of {total_slides}")
    print(f"Native tables       : {stats['tables']}")
    print(f"Native charts       : {stats['charts']}")
    print(f"Native equations    : {stats['equations']}")
    print(f"Pictures processed  : {stats['pictures']}  (failed/skipped: {stats['failed']})")
    print(f"Image regions OCR'd : {stats['image_regions_ocrd']}")
    print(f"Total time          : {elapsed:.1f}s")
    if ocr_times:
        avg = sum(ocr_times) / len(ocr_times)
        all_pics = sum(1 for s in prs.slides for sh in iter_shapes(s.shapes)
                       if not (getattr(sh, "has_chart", False) and sh.has_chart)
                       and is_candidate_picture(sh))
        print(f"Avg per picture     : {avg:.1f}s")
        print(f"Full deck estimate  : {all_pics} pictures x {avg:.1f}s = ~{all_pics * avg / 60:.1f} min")
    print(f"Output folder       : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()