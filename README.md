# Slide RAG (local)

Ask questions about your lecture slides. Everything runs on your own machine:
embeddings with sentence-transformers, keyword search with BM25, generation with llama-server.

```
slide_rag/
  rag.py        CLI: index / ask / chat / eval
  web.py        browser chat with the retrieved slides shown as pictures
  config.py     all paths, model names and knobs
  embed.py      text -> vectors (sentence-transformers)
  store.py      load chunks, build index, hybrid search, cited-slide expansion
  llm.py        prompt + llama-server call
  data/
    study_guide_chunks.jsonl        from Claude
    supplement_chunks.jsonl         9 topics NOT in the slides (TF-IDF, beam search, HMM/Viterbi,
                                    skip-gram, Naive Bayes, n-grams, perplexity, RAG, bag-of-words); delete to exclude
    slides/*.jsonl                  chunks.jsonl from slide_rag_test.py, one per lecture
    slide_images/                   Lecture01-01.jpg ... one picture per slide (for web.py)
  index/                            created by `python rag.py index`
  questions.txt                     your test questions for `eval`
```

## Setup (in your `rag` conda env)

```
pip install sentence-transformers rank_bm25 requests numpy
```

Put your files in `data/`:
- `data/study_guide_chunks.jsonl`
- `data/slides/Lecture01_chunks.jsonl`, `Lecture02_chunks.jsonl`, ... (rename each `chunks.jsonl`
  your pipeline produced; the `source_file` inside must still be the original .pptx name,
  e.g. `Lecture01-Introduction_HEU-NLP.pptx`, since that is how guide and slides are linked)

## Run

1. Build the index (downloads the 470 MB embedding model the first time):
   ```
   python rag.py index
   ```
2. Start a chat LLM with llama-server on port 8091, e.g.
   ```
   llama-server -m Qwen3-4B-Instruct-2507-Q4_K_M.gguf -c 8192 -ngl 99 --port 8091
   ```
   Any instruction-tuned GGUF works; ~4B at Q4 fits a 6 GB card with 8k context.
   (Stop the PaddleOCR-VL server first if VRAM is tight.)
3. Ask:
   ```
   python rag.py ask "How does logistic regression handle outliers compared to least squares?"
   python rag.py chat
   ```

## Check retrieval before trusting answers

`eval` prints which chunks are retrieved for each question, without calling the LLM:
```
python rag.py eval questions.txt
```
If the right slides do not appear, tune `TOP_K`, `EXPAND_CITED_SLIDES` or try another
embedding model in `config.py`, then rebuild the index.

## How a question is answered

1. The question is embedded (dense) and tokenized (BM25); both rankings are fused (RRF).
2. Top `TOP_K` chunks are kept. For each study-guide chunk among them, up to
   `EXPAND_CITED_SLIDES` of the slide chunks it cites are added, so the lecturer's
   original wording travels with the rewritten explanation.
3. Chunks are trimmed to `MAX_CONTEXT_CHARS` and sent to the LLM with their tags
   (`[L03 Slide 12]`, `[L03 Guide 3.5]`); the LLM is told to answer only from them and cite tags.

## Evaluate

Two question sets:
- `eval_set.jsonl` (default): 54 questions. 45 should be answered: exam-style questions modelled on
  the sample exam (new examples, small calculations, design scenarios), including the RAG questions
  (answered from Extra X8), and one question per supplementary topic. 9 are covered by no source: u1–u5 (off-topic, or a
  detail the lecturer never gave) and r1–r4 (NLP topics covered nowhere). For these the model must
  either refuse or answer under a "[General knowledge]" tag; answering from memory untagged is a failure.
- `eval_set_concepts.jsonl`: 42 questions that use the slides' own wording (easier for retrieval).

```
python eval.py retrieval            # no LLM needed, takes seconds
python eval.py generation           # needs llama-server; answers every question
python eval.py retrieval eval_set_concepts.jsonl   # the other question set
python eval.py generation --judge   # also asks the LLM whether each answer is supported
```

**Retrieval** (per mode: dense, bm25, hybrid, and the final context the LLM sees)
- `hit@k`: a gold slide (or a guide chunk citing it) is in the top k
- `recall`: share of the gold slides covered by the retrieved chunks
- `mrr`: 1 / rank of the first relevant chunk (1.0 = always first)

**Generation**
- `fact recall` / `all facts present`: key facts found in the answer (string match, so check misses by hand)
- `cites a gold source`: the answer cites at least one relevant chunk
- `citations valid`: cited tags were really given to the model (catches made-up citations)
- `outside-source questions refused or tagged`: the 9 questions no source covers are either refused
  or answered under "[General knowledge]"; `wrong refusal`: refuses an answerable one
- `judged supported` (optional): the local LLM grades its own answers. A small model is a lenient
  judge, so treat this as a rough signal only

Per-question details, including full answers, are written to `eval_results/`. Change one setting
at a time (e.g. TOP_K or EMBED_MODEL), rebuild the index if needed, and compare runs.

## Supplementary topics (not in the slides)

`data/supplement_chunks.jsonl` adds 9 standard NLP topics that Lectures 01–07 do not cover. They are
written from general textbook knowledge, not from your lecturer, so they are kept apart: they are
tagged `[Extra X1]` … `[Extra X9]` instead of slide tags, and the model is told to say
"(not from the lecture slides)" when it uses them. To check they do not crowd out the lecture
material, run `python eval.py retrieval` with and without the file (rebuild the index each time).

## Answers outside the sources

When the sources do not cover a question (or only part of it), the model answers from its own
knowledge under a "[General knowledge]" line, and the program adds a note that this part is not
from the course and may be wrong. Set `MIN_TOP_SCORE` in `config.py` to send clearly off-topic
questions straight to general knowledge without any course sources (faster, and always tagged), or
set `OFF_TOPIC = "refuse"` to refuse them instead.

## Browser chat (web.py)

```
pip install gradio pywin32
python web.py
```
A page opens at http://127.0.0.1:7860. Under each answer you see the slides it used: ★ marks slides
the answer cites, the others were retrieved. Click a slide to enlarge it, then "Open selected slide
in PowerPoint" opens the .pptx at that slide (needs PowerPoint, pywin32 and `PPTX_DIR` in `config.py`;
without pywin32 it opens the file at slide 1). llama-server must be running as for `rag.py`.
Each question is answered on its own: the model does not see earlier questions in the chat.
