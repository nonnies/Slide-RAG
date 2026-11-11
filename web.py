"""Browser chat for the slide RAG, with the retrieved slides shown under each answer.

  pip install gradio
  python web.py                 then open http://127.0.0.1:7860

Needs llama-server running (config.LLM_URL), the index (python rag.py index),
slide images in config.SLIDE_IMAGE_DIR, and config.PPTX_DIR set to open slides in PowerPoint.
"""
import inspect
import os
import sys

import gradio as gr
import requests

import config
from embed import Embedder
from eval import guide_slide_map, parse_citations
from llm import source_tag
from rag import stream_answer
from store import Store, lecture_key

MAX_SLIDES_SHOWN = 12

store, embedder = Store(), Embedder()
GUIDE_SLIDES = guide_slide_map(store)                      # ("L05", "5.6") -> {30, 31, ...}
DECKS = {}                                                 # "L05" -> "Lecture05-Neural_Networks_II.pptx"
for c in store.chunks:
    src = c["metadata"].get("source_file")
    if src:
        DECKS.setdefault("L" + lecture_key(src)[-2:], src)


# ---------------- which slides to show --------------------------------------

def slide_image(lec, n):
    path = config.SLIDE_IMAGE_DIR / f"Lecture{lec[1:]}-{n:02d}.jpg"
    return path if path.exists() else None


def slides_for(answer, chunks):
    """[(lec, slide, cited?)]: slides the answer cites first, then other retrieved slides."""
    out = []
    for lec, kind, num in parse_citations(answer):
        if kind == "slide":
            out.append((lec, num, True))
        elif kind == "guide":                              # a guide section stands for the slides it cites
            out += [(lec, n, True) for n in sorted(GUIDE_SLIDES.get((lec, num), ()))]
    for c in chunks:
        m = c["metadata"]
        if m["layer"] == "slides":
            out.append(("L" + lecture_key(m["source_file"])[-2:], m["slide"], False))
    seen, result = set(), []
    for lec, n, cited in out:
        if (lec, n) not in seen and lec in DECKS:
            seen.add((lec, n))
            result.append((lec, n, cited))
    return result[:MAX_SLIDES_SHOWN]


def gallery_items(slides):
    items = []
    for lec, n, cited in slides:
        img = slide_image(lec, n)
        if img:
            items.append((str(img), f"{'★ ' if cited else ''}{lec} Slide {n}"))
    return items


def other_sources(chunks):
    """Sources without a slide picture: supplementary notes."""
    tags = [source_tag(c) for c in chunks if c["metadata"]["layer"] == "supplement"]
    return "Also used: " + ", ".join(dict.fromkeys(tags)) + " (textbook notes, not slides)" if tags else ""


# ---------------- opening a slide in PowerPoint ---------------------------------

def open_in_powerpoint(lec, n):
    path = config.PPTX_DIR / DECKS[lec]
    if not path.exists():
        return f"Not found: {path}. Set PPTX_DIR in config.py."
    if sys.platform != "win32":
        return "Opening PowerPoint only works on Windows."
    try:                                                   # PowerPoint: open the deck at that slide
        import win32com.client
        app = win32com.client.Dispatch("PowerPoint.Application")
        app.Visible = True
        pres = next((p for p in app.Presentations if os.path.samefile(p.FullName, path)), None)
        pres = pres or app.Presentations.Open(str(path))
        pres.Windows(1).Activate()
        pres.Windows(1).View.GotoSlide(n)
        return f"Opened {DECKS[lec]} at slide {n}."
    except Exception:                                      # no PowerPoint/pywin32: default app, slide 1
        os.startfile(str(path))
        return f"Opened {DECKS[lec]} (go to slide {n} yourself; install pywin32 to jump there)."


# ---------------- the page --------------------------------------------------

def respond(question, history):
    history = history + [{"role": "user", "content": question},
                         {"role": "assistant", "content": ""}]
    yield history, [], [], "", ""
    try:
        for text, chunks in stream_answer(store, embedder, question):
            history[-1]["content"] = text
            yield history, gr.skip(), gr.skip(), gr.skip(), gr.skip()
    except requests.RequestException as e:
        history[-1]["content"] = f"Could not reach the model at {config.LLM_URL}. Is llama-server running?\n\n{e}"
        yield history, [], [], "", ""
        return
    slides = slides_for(text, chunks)
    shown = [s for s in slides if slide_image(s[0], s[1])]
    note = other_sources(chunks)
    if chunks and not shown:
        note = (note + "  " if note else "") + "(No slide images found in " + str(config.SLIDE_IMAGE_DIR) + ")"
    yield history, gallery_items(shown), shown, note, ""


def pick(shown, evt: gr.SelectData):
    lec, n, _ = shown[evt.index]
    return (lec, n), f"Selected: {lec} Slide {n}"


def open_selected(selected):
    return open_in_powerpoint(*selected) if selected else "Click a slide first."


with gr.Blocks(title="NLP slides assistant") as demo:
    gr.Markdown("## NLP lecture assistant\nAnswers come from your slides and study guide. "
                "★ = slide cited in the answer. Click a slide to enlarge it, then open it in PowerPoint.")
    # Gradio 5 needs type="messages"; Gradio 6 only has that format and dropped the argument
    chat = gr.Chatbot(height=480, **({"type": "messages"}
                                     if "type" in inspect.signature(gr.Chatbot).parameters else {}))
    box = gr.Textbox(placeholder="Ask about the lectures and press Enter", show_label=False)
    gr.Markdown("**Slides for the latest answer**")
    gallery = gr.Gallery(columns=4, height="auto", object_fit="contain", show_label=False)
    with gr.Row():
        status = gr.Markdown("")
        open_btn = gr.Button("Open selected slide in PowerPoint", scale=0)
    extra = gr.Markdown("")
    shown_state, selected_state = gr.State([]), gr.State(None)

    box.submit(respond, [box, chat], [chat, gallery, shown_state, extra, box])
    gallery.select(pick, shown_state, [selected_state, status])
    open_btn.click(open_selected, selected_state, status)
    gr.Button("Clear").click(lambda: ([], [], [], "", "", None),
                             None, [chat, gallery, shown_state, extra, status, selected_state])

if __name__ == "__main__":
    demo.launch(allowed_paths=[str(config.SLIDE_IMAGE_DIR)], inbrowser=True)