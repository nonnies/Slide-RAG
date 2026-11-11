"""Evaluate retrieval and generation against a gold question set (eval_set.jsonl).

  python eval.py retrieval            no LLM needed; compares dense / BM25 / hybrid / final context
  python eval.py generation           needs llama-server; answers every question and scores it
  python eval.py generation --judge   also asks the LLM to judge if each answer is supported
  python eval.py all                  both
  python eval.py retrieval eval_set_concepts.jsonl   use another question file (default: eval_set.jsonl)

Gold set format, one JSON object per line:
  {"id", "question", "lecture": "L03", "slides": [20], "facts": ["linearly separable", "x1x2|product"]}
  {"id", "question", "unanswerable": true}          # off-topic: answer should say the sources don't cover it
  {"id", "question", "in_slides": false}            # exam topic the slides don't cover: same expectation
Optional "area" (e.g. "2 RNN") groups the scores by exam area.
A fact may list alternatives separated by "|"; it counts if any alternative appears in the answer.

Results: printed summary + per-question details in eval_results/*.jsonl.
"""
import json
import re
import sys
import time
from pathlib import Path

import config
from embed import Embedder
from llm import ask_llm, build_messages, source_tag
from store import Store, lecture_key

EVAL_SET = config.ROOT / "eval_set.jsonl"
OUT_DIR = config.ROOT / "eval_results"
KS = (1, 3, 5)                                       # cut-offs for Hit@k

ABSTAIN_PHRASES = ("not contain", "does not include", "doesn't include", "not mention",
                   "no information", "not covered", "not provide", "not in the sources",
                   "cannot answer", "can't answer", "cannot be answered", "unable to answer",
                   "do not cover", "does not cover", "don't cover", "doesn't cover", "not discussed",
                   "not specified", "no mention")


# ---------------- helpers ----------------------------------------------------

def load_eval_set(path=EVAL_SET):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def no_answer_expected(q):
    return q.get("unanswerable") or q.get("in_slides") is False


def by_area(rows, metric):
    """{area: mean of metric} for rows that have the metric, in area order."""
    groups = {}
    for r in rows:
        if r.get(metric) is not None:
            groups.setdefault(r.get("area", "-"), []).append(r[metric])
    return {a: (mean(v), len(v)) for a, v in sorted(groups.items())}


def norm(text):
    """Lowercase and drop spaces/underscores/braces so 'W_{hh}' matches 'whh'."""
    return re.sub(r"[\s_{}$\\]", "", text.lower())


def chunk_lecture(chunk):
    m = chunk["metadata"]
    return m.get("lecture_id") or "L" + lecture_key(m.get("source_file", ""))[-2:]


def covered_slides(chunk, q):
    """Gold slides of question q that this chunk contains (slide chunk) or cites (guide chunk)."""
    if chunk_lecture(chunk) != q["lecture"]:
        return set()
    m = chunk["metadata"]
    mine = set(m.get("cited_slides", [])) if m["layer"] == "study_guide" else {m.get("slide")}
    return mine & set(q["slides"])


def save(name, rows):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ---------------- retrieval --------------------------------------------------

def retrieval_scores(ranked, q):
    """ranked: list of chunks in rank order. Returns hit@k, recall, reciprocal rank."""
    rel = [bool(covered_slides(c, q)) for c in ranked]
    first = next((i for i, r in enumerate(rel) if r), None)
    found = set().union(*(covered_slides(c, q) for c in ranked)) if ranked else set()
    out = {f"hit@{k}": float(any(rel[:k])) for k in KS}
    out["recall"] = len(found) / len(q["slides"])
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    return out


def eval_retrieval(store, embedder, questions):
    k = config.TOP_K
    modes = {
        "dense": lambda q, v: [store.chunks[i] for i, _ in store.dense(v, k)],
        "bm25": lambda q, v: [store.chunks[i] for i, _ in store.keyword(q, k)],
        "hybrid": lambda q, v: [store.chunks[i] for i, _ in store.hybrid(q, v, k)[0]],
        "context": lambda q, v: store.search(q, v),        # hybrid + cited-slide expansion + budget
    }
    rows, per_mode = [], {m: [] for m in modes}
    for q in questions:
        if no_answer_expected(q):
            continue
        qvec = embedder.query(q["question"])
        row = {"id": q["id"], "area": q.get("area", "-"), "question": q["question"],
               "gold": f"{q['lecture']} {q['slides']}"}
        for name, fn in modes.items():
            ranked = fn(q["question"], qvec)
            s = retrieval_scores(ranked, q)
            per_mode[name].append(s)
            row[name] = s
            if name == "context":
                row["retrieved"] = [source_tag(c) for c in ranked]
                row["context_hit@5"] = s["hit@5"]
        rows.append(row)

    print(f"\nRETRIEVAL  ({len(rows)} answerable questions, TOP_K={k})")
    cols = [f"hit@{k}" for k in KS] + ["recall", "mrr"]
    print(f"  {'mode':<9}" + "".join(f"{c:>9}" for c in cols))
    for name, scores in per_mode.items():
        print(f"  {name:<9}" + "".join(f"{mean([s[c] for s in scores]):>9.2f}" for c in cols))
    print("  (context = what the LLM actually sees: hybrid + cited-slide expansion. A guide chunk already")
    print("   counts for the slides it cites, so expansion cannot raise these scores; its effect shows")
    print("   up in generation, as answers citing the original slides.)")

    areas = by_area(rows, "context_hit@5")
    if len(areas) > 1:
        print("\n  final-context hit@5 by area:")
        for a, (v, n) in areas.items():
            print(f"    {a:<20} {v:5.2f}   ({n} questions)")

    misses = [r for r in rows if r["context"]["hit@5"] == 0]
    if misses:
        print(f"\n  Gold slide not in the top 5 of the final context ({len(misses)}):")
        for r in misses:
            print(f"    {r['id']} gold {r['gold']:<22} got {' '.join(r['retrieved'][:5])}")
            print(f"        {r['question']}")
    print(f"\n  details: {save('retrieval', rows)}")


# ---------------- generation -------------------------------------------------

CITE_RE = re.compile(r"L0?(\d{1,2})\s+(Slide|Guide)\s+([\d.]*\d)(?:\s+part\s+(\d+))?", re.I)


def cited_tags(answer):
    out = []
    for lec, kind, num, part in CITE_RE.findall(answer):
        tag = f"[L{int(lec):02d} {kind.capitalize()} {num}" + (f" part {part}" if part else "") + "]"
        out.append(tag)
    return list(dict.fromkeys(out))


JUDGE_PROMPT = """You check answers for a study assistant.
Below are SOURCES and an ANSWER. Decide whether every factual claim in the ANSWER is supported by the SOURCES.
Reply with exactly one word: SUPPORTED or UNSUPPORTED."""


def judge(answer, chunks):
    sources = "\n\n-----\n\n".join(f"{source_tag(c)}\n{c['text']}" for c in chunks)
    msgs = [{"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"SOURCES:\n\n{sources}\n\nANSWER:\n{answer}"}]
    verdict = ask_llm(msgs).strip().upper()
    return 0.0 if "UNSUPPORTED" in verdict else 1.0 if "SUPPORTED" in verdict else None


def eval_generation(store, embedder, questions, use_judge=False):
    rows = []
    for i, q in enumerate(questions, 1):
        chunks = store.search(q["question"], embedder.query(q["question"]))
        t0 = time.perf_counter()
        answer = ask_llm(build_messages(q["question"], chunks))
        seconds = time.perf_counter() - t0
        given = {source_tag(c) for c in chunks}
        cites = cited_tags(answer)
        a = norm(answer)
        abstained = any(p.replace(" ", "") in a for p in ABSTAIN_PHRASES)
        row = {"id": q["id"], "area": q.get("area", "-"), "question": q["question"],
               "answer": answer, "seconds": round(seconds, 1),
               "abstained": abstained, "citations": cites,
               "citations_valid": mean([c in given for c in cites]) if cites else None}
        if no_answer_expected(q):
            row["unanswerable"] = True
            row["kind"] = "not in slides" if q.get("in_slides") is False else "off-topic"
        else:
            hits = [f for f in q["facts"] if any(norm(alt) in a for alt in f.split("|"))]
            row["facts_found"] = hits
            row["facts_missing"] = [f for f in q["facts"] if f not in hits]
            row["fact_recall"] = len(hits) / len(q["facts"])
            gold_tags = {source_tag(c) for c in chunks if covered_slides(c, q)}
            row["cites_gold"] = float(bool(set(cites) & gold_tags))
        if use_judge and not row.get("unanswerable"):
            row["supported"] = judge(answer, chunks)
        rows.append(row)
        status = "abstain" if abstained else f"facts {row.get('fact_recall', 0):.0%}"
        print(f"  [{i}/{len(questions)}] {q['id']} {status:<12} {seconds:5.1f}s  {q['question'][:60]}")

    ans = [r for r in rows if not r.get("unanswerable")]
    una = [r for r in rows if r.get("unanswerable")]
    with_cites = [r for r in rows if r["citations_valid"] is not None]
    print(f"\nGENERATION  ({len(ans)} answerable + {len(una)} unanswerable questions)")
    print(f"  fact recall (avg share of key facts in answer) {mean([r['fact_recall'] for r in ans]):6.2f}")
    print(f"  all facts present                              {mean([r['fact_recall'] == 1 for r in ans]):6.2f}")
    print(f"  cites a gold source                            {mean([r['cites_gold'] for r in ans]):6.2f}")
    print(f"  citations point to given sources (valid)       {mean([r['citations_valid'] for r in with_cites]):6.2f}")
    print(f"  answers with no citation                       {mean([not r['citations'] for r in ans]):6.2f}")
    print(f"  correct abstention on unanswerable             {mean([r['abstained'] for r in una]):6.2f}")
    for r in una:
        if not r["abstained"]:
            print(f"      {r['id']} ({r['kind']}) answered anyway; check it is not invented")
    print(f"  wrong abstention on answerable                 {mean([r['abstained'] for r in ans]):6.2f}")
    if use_judge:
        judged = [r["supported"] for r in ans if r.get("supported") is not None]
        print(f"  judged supported by sources (LLM judge)        {mean(judged):6.2f}   ({len(judged)} judged)")
    print(f"  avg seconds per answer                         {mean([r['seconds'] for r in rows]):6.1f}")
    areas = by_area(ans, "fact_recall")
    if len(areas) > 1:
        print("\n  fact recall by area:")
        for a, (v, n) in areas.items():
            print(f"    {a:<20} {v:5.2f}   ({n} questions)")

    weak = [r for r in ans if r["fact_recall"] < 1 or not r["cites_gold"]]
    if weak:
        print(f"\n  Answers to check ({len(weak)}):")
        for r in weak:
            why = []
            if r["facts_missing"]:
                why.append("missing " + ", ".join(repr(f.split('|')[0]) for f in r["facts_missing"]))
            if not r["cites_gold"]:
                why.append("no gold citation")
            if r["abstained"]:
                why.append("abstained")
            print(f"    {r['id']}: {'; '.join(why)}")
    print(f"\n  details (full answers): {save('generation', rows)}")


# ---------------- main -------------------------------------------------------

def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd not in ("retrieval", "generation", "all"):
        print(__doc__)
        return
    files = [a for a in argv[2:] if not a.startswith("--")]
    questions = load_eval_set(Path(files[0]) if files else EVAL_SET)
    print(f"{len(questions)} questions from {files[0] if files else EVAL_SET.name}")
    store, embedder = Store(), Embedder()
    if cmd in ("retrieval", "all"):
        eval_retrieval(store, embedder, questions)
    if cmd in ("generation", "all"):
        print("\nGenerating answers...")
        eval_generation(store, embedder, questions, use_judge="--judge" in argv)


if __name__ == "__main__":
    main(sys.argv)