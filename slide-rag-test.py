"""
Slide -> RAG pipeline with typed content and contextual chunks

1. python-pptx reads NATIVE content exactly (no OCR), in reading order:
     title, headings, paragraphs, bullet lists, examples, tables, charts, formulas
2. Pictures go through the PaddleOCR-VL pipeline (layout detection picks the
   right prompt per region); regions labeled "image" are cropped and read with "OCR:".
3. EVERY piece of content is labeled with its type, e.g.
     [Title] ...   [Example] ...   [Table] ...   [Formula] ...   [Text in picture/diagram] ...
4. EVERY chunk starts with a context header:
     Lecture | Section | Slide n of N | Slide title | Previous slide | Contains: ...
5. Output:
     slide_XXX.md   human-readable, identical to what gets embedded
     chunks.jsonl   one JSON record per chunk (text + metadata) -> load into your RAG

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
PPTX_PATH = r"pptx\Lecture06-Transformers.pptx"    # <-- change this
LECTURE_NAME = ""     
OUTPUT_DIR = Path("slide_rag_output")
MAX_SLIDES = 100                              # test on first 10 slides; set large for all
SERVER_BASE = "http://127.0.0.1:8090/v1"     # your llama-server
DEVICE = "gpu:0"                             # layout model device; "cpu" if VRAM is tight
MIN_WIDTH_CM = 3.0                           # skip small pictures (icons, logos)
MAX_SIDE_PX = 2000                           # downscale huge images for speed
IMAGE_LABELS = {"image", "figure"}           # region labels that get "OCR:" fallback
MIN_REGION_PX = 40                           # skip tiny image regions
CROP_PAD_PX = 8                              # padding around cropped regions
SAVE_LAYOUT_VIS = True                       # save images with layout boxes (for checking)
REQUEST_TIMEOUT = 120                        # seconds per OCR request
OCR_MAX_TOKENS = 1024                        # stop runaway generation (diagram labels can loop forever)
MAX_LINE_REPEATS = 2                         # keep a repeated line at most this many times
MIN_UNIQUE_RATIO = 0.3                       # drop OCR text that is mostly repetition
MAX_CHUNK_CHARS = 1500                       # longer slides are split; each part keeps the header
EXAMPLE_PREFIXES = ("example", "examples", "e.g.", "eg:", "for example", "for instance",
                    "例如", "例子", "ตัวอย่าง")
# ----------------------------------------

EMU_PER_CM = 360_000
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
CHAT_URL = SERVER_BASE + "/chat/completions"

# How each content type is labeled in the chunk text
TYPE_LABELS = {
    "title": "Title",
    "heading": "Heading",
    "paragraph": "Text",
    "list": "Bullet list",
    "example": "Example",
    "picture_text": "Text in picture/diagram",
    "chart": "Chart data",
    "table": "Table",
    "formula": "Formula",
    "caption": "Caption",
    "code": "Code/Algorithm",
    "footnote": "Footnote",
}


def element(etype, text, source):
    """One labeled piece of slide content."""
    return {"type": etype, "text": text.strip(), "source": source}


def mark_examples(el):
    """Paragraphs/lists that start like an example become type 'example'."""
    if el["type"] in ("paragraph", "list"):
        first = el["text"].lstrip("-• ").lower()
        if first.startswith(EXAMPLE_PREFIXES):
            el["type"] = "example"
    return el


# ================= native extraction (python-pptx) =================

def iter_shapes(shapes):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from iter_shapes(sh.shapes)
        else:
            yield sh


def reading_order(shapes):
    """Top-to-bottom, then left-to-right (python-pptx order is drawing order, not reading)."""
    return sorted(shapes, key=lambda s: ((s.top or 0) // 200_000, s.left or 0))


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


def text_frame_to_element(tf):
    """One paragraph -> 'paragraph'; several paragraphs -> indented 'list'."""
    paras = [(p.level, p.text.strip()) for p in tf.paragraphs if p.text.strip()]
    if not paras:
        return None
    if len(paras) == 1:
        return element("paragraph", paras[0][1], "native")
    return element("list", "\n".join(f"{'  ' * lvl}- {t}" for lvl, t in paras), "native")


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


# ================= OCR =================

def clean_repetition(text):
    """Remove looping output (e.g. 'Trm Trm Trm ...' from architecture diagrams)."""
    seen, lines = {}, []
    for ln in text.splitlines():
        key = ln.strip()
        if not key:
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= MAX_LINE_REPEATS:
            lines.append(ln)
    cleaned = "\n".join(lines)
    words = cleaned.split()
    if len(words) >= 20 and len(set(words)) / len(words) < MIN_UNIQUE_RATIO:
        return ""        # mostly the same few tokens repeated -> no useful text
    return cleaned


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
        "max_tokens": OCR_MAX_TOKENS,
    })
    r.raise_for_status()
    choice = r.json()["choices"][0]
    text = choice["message"]["content"].strip()
    if choice.get("finish_reason") == "length":
        print("   (OCR hit the token limit, likely a repetition loop; cleaning output)")
    return clean_repetition(text)


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


def map_ocr_label(label):
    """Map PaddleOCR layout labels to our content types (None = drop as noise)."""
    l = label
    if l in IMAGE_LABELS:
        return "picture_text"
    if l in ("header", "footer", "number", "page_number", "header_image", "footer_image", "seal"):
        return None
    if "caption" in l or ("title" in l and any(k in l for k in ("figure", "table", "chart"))):
        return "caption"
    if "title" in l:
        return "heading"
    if "table" in l:
        return "table"
    if "formula" in l or "equation" in l:
        return "formula"
    if "chart" in l:
        return "chart"
    if "algorithm" in l or "code" in l:
        return "code"
    if "footnote" in l or "reference" in l:
        return "footnote"
    if "list" in l:
        return "list"
    return "paragraph"


def ocr_picture(pipeline, img_path, out_dir):
    """Layout pipeline + OCR: fallback on image regions. Returns (elements, stats)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    stats = {"regions": 0, "image_regions_ocrd": 0, "labels": []}
    elements = []

    for res in pipeline.predict(str(img_path)):
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
            etype = map_ocr_label(b["label"])
            if etype is None:
                continue
            if etype == "picture_text":
                if not b["bbox"]:
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"][:4]]
                x1, y1 = max(0, x1 - CROP_PAD_PX), max(0, y1 - CROP_PAD_PX)
                x2, y2 = min(W, x2 + CROP_PAD_PX), min(H, y2 + CROP_PAD_PX)
                if x2 - x1 < MIN_REGION_PX or y2 - y1 < MIN_REGION_PX:
                    continue
                try:
                    text = direct_ocr(pil.crop((x1, y1, x2, y2)), "OCR:")
                except Exception as e:
                    print(f"   OCR failed on image region: {e}")
                    continue
                lines = [ln for ln in text.splitlines() if ln.strip() and ln.strip() not in known_text]
                if lines:
                    stats["image_regions_ocrd"] += 1
                    elements.append(element("picture_text", "\n".join(lines), "ocr"))
            elif b["content"]:
                elements.append(mark_examples(element(etype, b["content"], "ocr")))

    # Safety net: layout found nothing usable -> OCR the whole picture
    if not elements:
        try:
            text = direct_ocr(pil, "OCR:")
            if text:
                elements.append(element("picture_text", text, "ocr"))
        except Exception as e:
            print(f"   whole-picture OCR failed: {e}")

    return elements, stats


# ================= slide processing =================

def process_slide(slide, n, pipeline, img_dir, stats, log, ocr_times):
    """Return {title, layout, elements, images} for one slide, in reading order."""
    title, elements, images = "", [], []
    pic_i = 0
    shapes = reading_order(list(iter_shapes(slide.shapes)))

    for sh in shapes:
        # native chart
        if getattr(sh, "has_chart", False) and sh.has_chart:
            elements.append(element("chart", chart_to_md(sh.chart), "native"))
            stats["charts"] += 1
            continue
        # native table
        if getattr(sh, "has_table", False) and sh.has_table:
            elements.append(element("table", table_to_md(sh.table), "native"))
            stats["tables"] += 1
            continue
        # native text (title / subtitle / body)
        if getattr(sh, "has_text_frame", False) and sh.has_text_frame and sh.text_frame.text.strip():
            ptype = str(sh.placeholder_format.type) if sh.is_placeholder else ""
            text = sh.text_frame.text.strip()
            if "SUBTITLE" in ptype:
                elements.append(element("heading", text, "native"))
            elif "TITLE" in ptype:
                if not title:
                    title = text
                elements.append(element("title", text, "native"))
            else:
                el = text_frame_to_element(sh.text_frame)
                if el:
                    elements.append(mark_examples(el))
            continue
        # picture -> OCR, inserted where the picture sits on the slide
        if is_candidate_picture(sh):
            pic_i += 1
            img = get_picture(sh)
            img_path = img_dir / f"slide_{n:03d}_pic_{pic_i:02d}.png"
            try:
                size = to_png(img.blob, img_path)
            except Exception as e:
                print(f"[slide {n}] picture {pic_i} skipped ({img.ext}): {e}")
                stats["failed"] += 1
                continue
            images.append(img_path.name)
            t0 = time.perf_counter()
            try:
                pic_elements, s = ocr_picture(pipeline, img_path,
                                              OUTPUT_DIR / "ocr_raw" / img_path.stem)
            except Exception as e:
                print(f"[slide {n}] picture {pic_i} OCR failed: {e}")
                pic_elements, s = [], {"regions": 0, "image_regions_ocrd": 0, "labels": []}
                stats["failed"] += 1
            dt = time.perf_counter() - t0
            ocr_times.append(dt)
            stats["pictures"] += 1
            stats["image_regions_ocrd"] += s["image_regions_ocrd"]
            log.append({"slide": n, "picture": img_path.name, "seconds": round(dt, 2),
                        "labels": s["labels"], "image_regions_ocrd": s["image_regions_ocrd"]})
            print(f"[slide {n}] picture {pic_i} ({size[0]}x{size[1]}) {dt:.1f}s | "
                  f"regions: {s['labels']}")
            elements.extend(pic_elements)

    # native equations (Office Math)
    for eq in equations_in_slide(slide):
        elements.append(element("formula", eq, "native"))
        stats["equations"] += 1

    # slides that are one big picture have no native title -> take it from OCR
    if not title:
        for el in elements:
            if el["type"] in ("title", "heading"):
                title = el["text"].splitlines()[0]
                break
    if not title:
        for el in elements:
            if el["text"]:
                title = el["text"].lstrip("-• ").splitlines()[0][:100]
                break

    layout = (slide.slide_layout.name or "").lower()
    return {"title": title, "layout": layout, "elements": elements, "images": images}


# ================= chunk building =================

def render_element(el):
    label = TYPE_LABELS.get(el["type"], "Text")
    if "\n" in el["text"] or el["type"] in ("table", "chart"):
        return f"[{label}]\n{el['text']}"
    return f"[{label}] {el['text']}"


def build_header(lecture, section, n, total, title, prev_title, types):
    lines = [f"Lecture: {lecture}"]
    if section:
        lines.append(f"Section: {section}")
    lines.append(f"Slide {n} of {total}: {title or '(untitled)'}")
    if prev_title:
        lines.append(f"Previous slide: {prev_title}")
    if types:
        lines.append("Contains: " + ", ".join(TYPE_LABELS[t].lower() for t in types))
    return "\n".join(lines)


def split_elements(rendered, limit):
    """Group rendered elements into parts under the size limit (never split one element)."""
    parts, cur, size = [], [], 0
    for r in rendered:
        if cur and size + len(r) > limit:
            parts.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += len(r) + 2
    if cur:
        parts.append(cur)
    return parts


def build_chunks(slides, lecture, source_file, total):
    chunks, section = [], ""
    for idx, s in enumerate(slides):
        n = s["n"]
        # section-header slides start a new section
        if "section" in s["layout"] and s["title"]:
            section = s["title"]
        prev_title = slides[idx - 1]["title"] if idx > 0 else ""
        next_title = slides[idx + 1]["title"] if idx + 1 < len(slides) else ""

        body = [e for e in s["elements"] if e["text"] and not (e["type"] == "title" and e["text"] == s["title"])]
        if not body:            # title-only / section slides: used for context, not stored as chunks
            continue
        types = list(dict.fromkeys(e["type"] for e in body if e["type"] not in ("paragraph", "list")))
        header = build_header(lecture, section, n, total, s["title"], prev_title, types)
        rendered = [render_element(e) for e in body] or ["(no text content on this slide)"]
        parts = split_elements(rendered, MAX_CHUNK_CHARS)

        for p_i, part in enumerate(parts, 1):
            part_note = f" (part {p_i} of {len(parts)})" if len(parts) > 1 else ""
            text = header + part_note + "\n\n" + "\n\n".join(part)
            chunks.append({
                "id": f"{Path(source_file).stem}_s{n:03d}_p{p_i}",
                "text": text,
                "metadata": {
                    "lecture": lecture,
                    "source_file": Path(source_file).name,
                    "section": section,
                    "slide": n,
                    "part": p_i,
                    "parts": len(parts),
                    "title": s["title"],
                    "prev_title": prev_title,
                    "next_title": next_title,
                    "content_types": list(dict.fromkeys(e["type"] for e in body)),
                    "images": s["images"],
                },
            })
    return chunks


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
    lecture = LECTURE_NAME or Path(PPTX_PATH).stem

    print("Loading PaddleOCR-VL pipeline (layout model + llama-server backend)...")
    t_load = time.perf_counter()
    pipeline = build_pipeline()
    print(f"Pipeline ready in {time.perf_counter() - t_load:.1f}s\n")

    ocr_times, t_start = [], time.perf_counter()
    stats = {"tables": 0, "charts": 0, "equations": 0, "pictures": 0,
             "image_regions_ocrd": 0, "failed": 0}
    log, slides = [], []

    # Pass 1: extract typed elements from every slide
    for n, slide in enumerate(prs.slides, 1):
        if n > n_slides:
            break
        data = process_slide(slide, n, pipeline, img_dir, stats, log, ocr_times)
        data["n"] = n
        slides.append(data)
        print(f"[slide {n}] {len(data['elements'])} elements | title: {data['title'][:60]!r}")

    # Pass 2: build chunks with context headers (needs neighbors' titles)
    chunks = build_chunks(slides, lecture, PPTX_PATH, total_slides)

    for s in slides:
        slide_chunks = [c["text"] for c in chunks if c["metadata"]["slide"] == s["n"]]
        (OUTPUT_DIR / f"slide_{s['n']:03d}.md").write_text(
            "\n\n---\n\n".join(slide_chunks), encoding="utf-8")

    with open(OUTPUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    (OUTPUT_DIR / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    type_counts = {}
    for s in slides:
        for e in s["elements"]:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1

    print("\n========== SUMMARY ==========")
    print(f"Slides processed    : {n_slides} of {total_slides}")
    print(f"Chunks written      : {len(chunks)}")
    print("Content by type     : " + ", ".join(f"{TYPE_LABELS.get(t, t)}={c}"
                                            for t, c in sorted(type_counts.items())))
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