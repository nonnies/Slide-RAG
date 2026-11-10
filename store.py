"""Load chunks, build/load the index, search it.

Index on disk = two files in INDEX_DIR:
  chunks.jsonl   every chunk (text + metadata), one per line, in index order
  vectors.npy    one normalized embedding per chunk, same order
"""
import json
import re
from pathlib import Path

import numpy as np

import config

_TOKEN = re.compile(r"[a-z0-9_]+|[一-鿿]")   # words, or single CJK characters


def tokenize(text):
    return _TOKEN.findall(text.lower())


def lecture_key(source_file):
    """'Lecture01-Introduction_HEU-NLP.pptx' -> 'lecture01' (links guide chunks to slide chunks)."""
    return Path(source_file).stem[:9].lower()


# ---------------- loading -----------------------------------------------

def load_chunks():
    """Read the study-guide chunks and all slide chunk files into one list."""
    chunks = []
    with open(config.STUDY_GUIDE_CHUNKS, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            c["metadata"]["layer"] = "study_guide"
            chunks.append(c)
    skipped = {"empty": 0, "outline": 0, "duplicate": 0}
    skip_title = re.compile(config.SKIP_TITLE_RE, re.I)
    seen_bodies = set()
    for path in config.SLIDE_CHUNK_FILES:
        with open(path, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                m = c["metadata"]
                body = c["text"].split("\n\n", 1)[-1].strip()   # text after the context header
                if len(body) < config.MIN_BODY_CHARS:            # title-only / logo / video slides
                    skipped["empty"] += 1
                    continue
                if skip_title.search((m.get("title") or "").strip()):
                    skipped["outline"] += 1
                    continue
                key = (lecture_key(m.get("source_file", "")), body)
                if config.SKIP_DUPLICATE_SLIDES and key in seen_bodies:   # repeated outline / recap
                    skipped["duplicate"] += 1
                    continue
                seen_bodies.add(key)
                m["layer"] = "slides"
                chunks.append(c)
    if any(skipped.values()):
        print("Skipped slide chunks: " + ", ".join(f"{v} {k}" for k, v in skipped.items() if v))
    if not chunks:
        raise SystemExit(f"No chunks found. Check paths in config.py (DATA_DIR={config.DATA_DIR})")
    return chunks


# ---------------- building ------------------------------------------------

def build_index(embedder):
    chunks = load_chunks()
    vectors = embedder.passages([c["text"] for c in chunks])
    config.INDEX_DIR.mkdir(exist_ok=True)
    with open(config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(config.INDEX_DIR / "vectors.npy", vectors)
    n_guide = sum(c["metadata"]["layer"] == "study_guide" for c in chunks)
    print(f"Indexed {len(chunks)} chunks ({n_guide} study guide, {len(chunks) - n_guide} slides) "
          f"-> {config.INDEX_DIR}")


# ---------------- searching -----------------------------------------------

class Store:
    def __init__(self):
        from rank_bm25 import BM25Okapi
        path = config.INDEX_DIR / "chunks.jsonl"
        if not path.exists():
            raise SystemExit("Index not found. Run:  python rag.py index")
        self.chunks = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.vectors = np.load(config.INDEX_DIR / "vectors.npy")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        # (lecture, slide number) -> index of that slide chunk, for citation expansion
        self.slide_lookup = {}
        for i, c in enumerate(self.chunks):
            m = c["metadata"]
            if m["layer"] == "slides":
                self.slide_lookup.setdefault((lecture_key(m["source_file"]), m["slide"]), i)

    def dense(self, qvec, k):
        scores = self.vectors @ qvec
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    def keyword(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def hybrid(self, query, qvec, k):
        """Reciprocal rank fusion of dense + keyword rankings."""
        fused = {}
        for ranking in (self.dense(qvec, config.DENSE_K), self.keyword(query, config.BM25_K)):
            for rank, (i, _) in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        ranked = sorted(fused.items(), key=lambda x: -x[1])
        return ranked[:k], fused

    def expand(self, hits, scores):
        """Append slide chunks cited by retrieved study-guide chunks (best-scoring ones first)."""
        chosen = [i for i, _ in hits]
        extra = []
        for i in chosen:
            m = self.chunks[i]["metadata"]
            if m["layer"] != "study_guide":
                continue
            cited = [self.slide_lookup.get((lecture_key(m["source_file"]), n))
                     for n in m.get("cited_slides", [])]
            cited = [j for j in cited if j is not None and j not in chosen and j not in extra]
            cited.sort(key=lambda j: -scores.get(j, 0.0))      # query-relevant slides first
            extra.extend(cited[:config.EXPAND_CITED_SLIDES])
        return chosen + extra

    def search(self, query, qvec):
        hits, scores = self.hybrid(query, qvec, config.TOP_K)
        ids = self.expand(hits, scores)
        # trim to the context budget
        out, used = [], 0
        for i in ids:
            c = self.chunks[i]
            if used + len(c["text"]) > config.MAX_CONTEXT_CHARS:
                continue
            out.append(c)
            used += len(c["text"])
        return out