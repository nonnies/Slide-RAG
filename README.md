# Slide RAG (local)

Ask questions about your lecture slides. Everything runs on your own machine:
embeddings with sentence-transformers, keyword search with BM25, generation with llama-server.

```
slide_rag/
  rag.py        CLI: index / ask / chat / eval
  config.py     all paths, model names and knobs
  embed.py      text -> vectors (sentence-transformers)
  store.py      load chunks, build index, hybrid search, cited-slide expansion
  llm.py        prompt + llama-server call
  data/
    study_guide_chunks.jsonl        from Claude
    slides/*.jsonl                  chunks.jsonl from slide_rag_test.py, one per lecture
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
