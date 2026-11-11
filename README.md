# Slide RAG (local)

A fully local study assistant for a university course's lecture slides. It reads the `.pptx`
decks (text, tables, formulas and the text inside pictures), indexes them together with a study
guide, and answers questions with a local LLM. Every answer cites the slides it came from, and a
browser page shows those slides and opens them in PowerPoint at the right page.

Nothing leaves your computer: OCR, embeddings, search and the LLM all run locally.

> **Course material is not included.** The slides, OCR output, slide images and study guide belong
> to the course and are kept out of this repository (see `.gitignore`). Bring your own decks and
> follow the steps below.


```
.pptx decks ──► 1. OCR + chunking ─────────────┐
                (slide-rag-OCR.py,             │
                 PaddleOCR-VL in llama.cpp)    ├─► 4. index ──► 5. ask ──► answer with
study guide  ──► 2. guide chunks ──────────────┤   (e5 +        (hybrid      citations,
(long-context LLM, cites slides)               │    BM25)        search +    slide pictures,
supplementary notes ─► 3. supplement chunks ───┘                 local LLM)  PowerPoint link
```

## Contents

- [Hardware and software](#hardware-and-software)
- [Project layout](#project-layout)
- [Step 1: OCR the slides into chunks](#step-1-ocr-the-slides-into-chunks)
- [Step 2: study guide chunks](#step-2-study-guide-chunks)
- [Step 3: supplementary notes (optional)](#step-3-supplementary-notes-optional)
- [Step 4: build the index](#step-4-build-the-index)
- [Step 5: start the chat model](#step-5-start-the-chat-model)
- [Step 6: ask questions](#step-6-ask-questions)
- [How an answer is produced](#how-an-answer-is-produced)
- [Evaluation](#evaluation)
- [Results](#results)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

## Hardware and software

Developed and tested on:
- Windows laptop, NVIDIA GTX 1660 Ti Max-Q (6 GB VRAM), AMD Ryzen 7 4800HS
- [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (CUDA build)
- Python 3.12 in conda environments
- Microsoft PowerPoint (only for the "open slide" button) and LibreOffice (only for rendering slide pictures)

Two Python environments keep the heavy OCR dependencies apart:

| Environment | Used for | Install |
|---|---|---|
| `ocr` | step 1 | see [step 1](#step-1-ocr-the-slides-into-chunks) |
| `rag` | steps 4–6 and evaluation | `pip install -r requirements.txt` |

## Project layout

```
slide-rag-OCR.py     step 1: .pptx -> typed, contextual chunks (python-pptx + PaddleOCR-VL)
rag.py               CLI: index / ask / chat / eval (retrieval only)
web.py               browser chat with slide pictures and "open in PowerPoint"
render_slides.py     .pptx -> one JPG per slide (for web.py)
config.py            all paths, models and settings
embed.py             text -> vectors (sentence-transformers)
store.py             load chunks, build the index, hybrid search, relevance check
llm.py               prompts, llama-server calls, labels for non-lecture material
eval.py              retrieval and generation evaluation
eval_set.jsonl            54 exam-style test questions
eval_set_concepts.jsonl   42 concept questions (held out: never used for tuning)
requirements.txt

data/                (not in git: your course material)
  slides/*.jsonl              slide chunks from step 1, one file per lecture
  study_guide_chunks.jsonl    step 2
  supplement_chunks.jsonl     step 3 (optional)
  slide_images/               Lecture01-01.jpg ... from render_slides.py
index/               (not in git) built by `python rag.py index`
eval_results/        (not in git) written by eval.py
```

## Step 1: OCR the slides into chunks

`slide-rag-OCR.py` turns every deck into chunks that keep the slide's structure.

1. **Native content first.** `python-pptx` reads titles, headings, paragraphs, bullet lists, tables,
   charts and equations exactly, in reading order. It also handles group shapes (child coordinate
   spaces) and equation text boxes hidden inside `mc:AlternateContent`.
2. **Pictures go through OCR.** Pictures wider than 3 cm are sent to the PaddleOCR-VL pipeline. A
   layout model (PP-DocLayoutV3) finds the regions, and each one is read with the right prompt
   (`OCR:`, `Table Recognition:`, `Chart Recognition:`, `Formula Recognition:`). Regions labelled
   as images or figures are cropped and read with `OCR:`, so text inside diagrams is kept.
   Runaway repetition is capped (`OCR_MAX_TOKENS`) and cleaned.
3. **Typed, contextual chunks.** Every piece is labelled (`[Title]`, `[Bullet list]`, `[Example]`,
   `[Table]`, `[Formula]`, `[Text in picture/diagram]` ...). Every chunk starts with a header:
   lecture, section, "Slide n of N", slide title, previous slide's title, and the content types it
   contains. Slides longer than `MAX_CHUNK_CHARS` are split into parts that each keep the header.

**OCR environment** (separate from the `rag` environment):
```
conda create -n ocr python=3.12
conda activate ocr
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]>=3.6.0"
python -m pip install python-pptx pillow requests certifi
```

**OCR model server.** Download `PaddleOCR-VL-1.6-Q8_0.gguf` and its `mmproj` file from
[PaddlePaddle/PaddleOCR-VL-1.6-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF), then:
```
llama-server -m PaddleOCR-VL-1.6-Q8_0.gguf --mmproj PaddleOCR-VL-1.6-mmproj.gguf ^
  --port 8090 --temp 0 --jinja -c 8192 -np 1 -ngl 99
```
The `--mmproj` file is the vision encoder; without it the model cannot see images.

**Run.** Set `PPTX_DIR` at the top of `slide-rag-OCR.py`, then:
```
python slide-rag-OCR.py
```
It processes every deck in `PPTX_DIR`. `RESUME = True` skips decks that are already done, so you
can rerun after a crash. Output:
```
slide_rag_output/
  <deck name>/slide_001.md ...   human-readable, identical to what gets embedded
  <deck name>/chunks.jsonl
  all_chunks/<deck name>_chunks.jsonl
```
Copy the files from `all_chunks/` into `data/slides/`.

Each line of a chunk file:
```json
{"id": "Lecture05-Neural_Networks_II_s012_p1",
 "text": "Lecture: Lecture05-Neural_Networks_II\nSlide 12 of 44: <slide title>\nPrevious slide: <title>\nContains: formula\n\n[Text] ...\n\n[Formula] ...",
 "metadata": {"lecture": "Lecture05-Neural_Networks_II", "source_file": "Lecture05-Neural_Networks_II.pptx",
              "section": "", "slide": 12, "part": 1, "parts": 1, "title": "...",
              "prev_title": "...", "next_title": "...", "content_types": ["text", "formula"], "images": []}}
```
`source_file` must stay the original `.pptx` name: it links slides, guide and pictures.

## Step 2: study guide chunks

Slides are terse, so the index also contains a study guide: a long document that explains each
lecture in full sentences and cites the slides for every paragraph, e.g. `[L05: Slides 31–32]`.
Ours was written by a long-context LLM (Claude) that read all the decks at once. Explanations that
are not on the slides are marked *[added explanation]*. A coverage check confirmed that every
slide is cited at least once.

The guide is split by section into `data/study_guide_chunks.jsonl`:
```json
{"id": "L05_sec5.6_p1",
 "text": "Course: ...\nLecture: Lecture 05: ...\nSection: 5.6 Vanishing and exploding gradients\nSlides cited: 30, 31, 32, 33, 37\n\n...",
 "metadata": {"lecture": "...", "lecture_id": "L05", "source_file": "Lecture05-Neural_Networks_II.pptx",
              "section": "5.6 Vanishing and exploding gradients", "cited_slides": [30, 31, 32, 33, 37]}}
```
`cited_slides` lets search add the original slides to the context whenever a guide chunk is found.

## Step 3: supplementary notes (optional)

`data/supplement_chunks.jsonl` holds standard topics the exam may ask about but the slides do not
cover. Ours has nine: TF-IDF, beam search, HMM/Viterbi, skip-gram with negative sampling, Naive
Bayes, n-gram language models, perplexity, retrieval-augmented generation and bag-of-words.
They are tagged `[Extra X1]` ... `[Extra X9]` and never mixed up with slides:
```json
{"id": "extra_X9_1", "text": "Supplementary topic: NOT from the lecture slides ...",
 "metadata": {"layer": "supplement", "topic_id": "X9", "title": "Bag-of-words representation",
              "related_slides": ["L02 Slide 3", "L01 Slide 44"]}}
```
Keep each chunk under about 2,000 characters: the embedding model only reads the first ~512 tokens.
Delete the file to leave the supplement out.

## Step 4: build the index

```
conda create -n rag python=3.12
conda activate rag
pip install -r requirements.txt
python rag.py index
```
The first run downloads the embedding model (`intfloat/multilingual-e5-small`). Indexing prints
which slide chunks it skipped and why:
- listed in `SKIP_SLIDES` (title, outline and video slides)
- empty (under 60 characters of content)
- title matches `SKIP_TITLE_RE` (outline, agenda, review, thanks ...)
- the same text as an earlier slide

Rebuild the index whenever the chunk files, `EMBED_MODEL` or the skip settings change. Prompt and
model changes do not need a rebuild.

## Step 5: start the chat model

Any instruction-tuned GGUF works. We use **Qwen3.5-9B** (4-bit, about 5.3 GB) with thinking turned off:
```
llama-server -m Qwen3.5-9B-<quant>.gguf --port 8090 --jinja -c 6144 -np 1 -ngl 99 ^
  -fa on -ctk q8_0 -ctv q8_0 -t 8 --reasoning-budget 0
```

| Flag | Why |
|---|---|
| `--port 8090` | where `config.LLM_URL` points (the OCR server uses 8090) |
| `--jinja` | use the model's own chat template |
| `-c 6144` | prompt (~3k tokens of sources) + answer (up to 1,024 tokens) with margin |
| `-ngl 99` | all layers on the GPU; lower it if you run out of VRAM |
| `-fa on -ctk q8_0 -ctv q8_0` | flash attention and an 8-bit KV cache: half the cache memory |
| `--reasoning-budget 0` | no hidden "thinking" before the answer (also sent per request via `LLM_EXTRA`) |

Stop the OCR server first if VRAM is tight. On Windows, check Task Manager → GPU → "Shared GPU
memory": if it grows while answering, the model spilled out of VRAM and runs slower than with
fewer GPU layers.

## Step 6: ask questions

**Browser** (recommended):
```
python render_slides.py      # once: one JPG per slide in data/slide_images/ (needs LibreOffice)
python web.py                # opens http://127.0.0.1:7860
```
Set `PPTX_DIR` in `config.py` first. Under each answer you see the slides behind it: ★ marks
slides the answer cites, the others were retrieved. Click a slide to enlarge it, then
**Open selected slide in PowerPoint** opens the deck at that slide (needs PowerPoint and `pywin32`;
without them the deck opens at slide 1).

**Terminal:**
```
python rag.py ask "Why does a Transformer scale attention scores by sqrt(D)?"
python rag.py chat
python rag.py eval questions.txt     # show what is retrieved, without the LLM
```

Each question is answered on its own: the model does not see earlier questions.

## How an answer is produced

1. **Relevance check.** The question is embedded. If even the best chunk scores below
   `MIN_TOP_SCORE` (cosine), the question is off-topic: it is answered from general knowledge
   without course sources, and labelled as such (or refused, with `OFF_TOPIC = "refuse"`).
2. **Hybrid search.** Dense search (e5 embeddings) and keyword search (BM25) each rank the chunks;
   reciprocal rank fusion merges the two rankings. The top `TOP_K` chunks are kept.
3. **Slide expansion.** For each guide or supplement chunk found, the best-matching slides it cites
   (`cited_slides` / `related_slides`) are added, so the lecturer's own wording reaches the model.
4. **Budget.** Chunks are added until `MAX_CONTEXT_CHARS` is reached.
5. **Prompt.** Each chunk is sent with a tag (`[L06 Slide 9]`, `[L06 Guide 6.4]`, `[Extra X1]`).
   The model must cite a tag after each claim and may apply the material to new examples and
   calculations. For anything no source covers, it writes a `[General knowledge]` line and answers
   from its own knowledge, without tags.
6. **Labels added by code.** Small models forget instructions, so the program itself appends a note
   when an answer cites `[Extra]` notes ("not from the lecture slides") or contains
   `[General knowledge]` ("written from memory, may be wrong: check it").

## Evaluation

```
python eval.py retrieval                         # seconds, no LLM needed
python eval.py generation                        # needs llama-server; ~35 s per question
python eval.py generation eval_set_concepts.jsonl
python eval.py rescore                           # re-score saved answers after changing the scoring
python eval.py generation --judge                # also lets the LLM grade its own answers (rough)
```

Each question in `eval_set.jsonl` names its gold slides (or supplement topic) and key facts:
```json
{"id": "s3", "area": "S Supervised & NN",
 "question": "A multiply gate computes x * y with x = 2 and y = -5. The upstream gradient is 3. What gradients flow back to x and to y?",
 "lecture": "L04", "slides": [49], "facts": ["-15|−15", "6"]}
```
Questions the sources do not cover have `"unanswerable": true` (off-topic) or `"in_slides": false`.

**Retrieval** is scored for dense, BM25, hybrid and the final context: hit@1/3/5 (a gold slide, or a
guide chunk citing it, in the top k), recall and MRR, plus hit@5 per exam area. It also prints the
best dense score of each question, which is how `MIN_TOP_SCORE` was chosen (the gap between the
lowest answerable question and clearly off-topic ones).

**Generation** reports:
- fact recall and "all facts present" (string match with `|` alternatives, so check misses by hand)
- cites a gold source
- citations valid: every cited tag was really given to the model (catches made-up citations)
- outside-source questions refused or tagged `[General knowledge]` (untagged answers from memory fail)
- wrong refusals, repetition loops, supplement labelling, seconds per answer

Every answer is saved to `eval_results/generation.jsonl`. Change one setting at a time and compare.

## Results

Exam-style set (54 questions: 45 answerable, 9 not covered by any source), temperature 0:

| | Qwen3.5-9B (4-bit) | Llama 3.1 8B (Q5_K_M) |
|---|---|---|
| Fact recall | **1.00** | 0.88 |
| All facts present | **1.00** | 0.82 |
| Cites a gold source | 0.98 | 0.98 |
| Citations valid | **1.00** | 0.99 |
| Outside-source refused or tagged | **1.00** | 0.89 |
| Seconds per answer | **35.0** | 45.9 |

Qwen with the relevance check (`MIN_TOP_SCORE = 0.80`), Llama without it. Llama's misses were mainly
calculations (bag-of-words vectors, softmax, gradients through a multiply gate) and one answer
from memory without the tag.

Retrieval on the same set (final context, TOP_K = 6): hit@5 **0.98**, recall 0.93, MRR 0.86
(dense alone 0.89, BM25 alone 0.91).

Held-out concept set (42 questions, never used for tuning), Qwen3.5-9B:
fact recall **0.99**, all facts 0.97, cites a gold source 1.00, citations valid 1.00,
outside-source handled 1.00.

The exam-style set was used while tuning prompts and settings, so its scores are optimistic; the
held-out set is the fairer number. Both sets are small and string-matched.

## Configuration reference

All settings live in `config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | embedding model (rebuild the index after changing) |
| `TOP_K` | 6 | chunks kept after hybrid search |
| `DENSE_K`, `BM25_K`, `RRF_K` | 20, 20, 60 | candidates per method and the fusion constant |
| `EXPAND_CITED_SLIDES` | 2 | slides added per guide or supplement chunk |
| `MAX_CONTEXT_CHARS` | 9000 | source text sent to the model |
| `SKIP_SLIDES`, `SKIP_TITLE_RE`, `MIN_BODY_CHARS` | | slides left out of the index |
| `LLM_URL` | `http://127.0.0.1:8091/...` | llama-server address |
| `LLM_TEMPERATURE` / `EVAL_TEMPERATURE` | 0.2 / 0.0 | chat / evaluation |
| `LLM_MAX_TOKENS` | 1024 | answer length limit |
| `LLM_REPEAT_PENALTY` | 1.1 | stops "0 0 0 ..." loops |
| `LLM_EXTRA` | `enable_thinking: false` | extra request fields (turns off Qwen thinking) |
| `MIN_TOP_SCORE` | None | relevance check; we use 0.80 (set it from `eval.py retrieval`) |
| `OFF_TOPIC` | `"general"` | off-topic questions: `"general"` (tagged answer) or `"refuse"` |
| `PPTX_DIR`, `SLIDE_IMAGE_DIR` | | decks and slide pictures for `web.py` |

## Troubleshooting

| Problem | Fix |
|---|---|
| OCR returns garbage characters | the model was sent a `.pptx`, not an image; the script handles this, but manual tests need a PNG/JPG |
| `FileNotFoundError` on an SSL certificate (conda) | `SSL_CERT_FILE` points to a missing file; the OCR script removes it, or copy `certifi`'s `cacert.pem` to the path in the error |
| OCR hangs on one picture | repetition loop; lower `OCR_MAX_TOKENS` or `REQUEST_TIMEOUT` |
| Chat prints nothing with Qwen | the model is thinking: keep `LLM_EXTRA` and add `--reasoning-budget 0` |
| `AttributeError: 'Store' object has no attribute 'relevant'` | old `store.py`; update all files together |
| Real course questions answered as `[General knowledge]` without sources | `MIN_TOP_SCORE` too high; lower it (e.g. 0.78) |
| "No slide images found" in the browser | files must be directly in `data/slide_images/`, not one folder deeper |
| PowerPoint opens at slide 1 | install `pywin32` in the `rag` env, restart `web.py`; unblock downloaded decks (Properties → Unblock) to avoid Protected View |
| `common_fit_params: ... n_gpu_layers already set by user` | harmless: llama.cpp skipped its automatic layer fitting because you set `-ngl` |

## Acknowledgements

- [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) for OCR, [llama.cpp](https://github.com/ggml-org/llama.cpp) for local inference
- [multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small) embeddings, [rank_bm25](https://github.com/dorianbrown/rank_bm25), [Gradio](https://www.gradio.app)
- The study guide, supplementary notes and parts of the code were written with help from Claude (Anthropic)