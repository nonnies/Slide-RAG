"""Evaluate retrieval and generation against a gold question set (eval_set.jsonl).

  python eval.py retrieval            no LLM needed; compares dense / BM25 / hybrid / final context
  python eval.py generation           needs llama-server; answers every question and scores it
  python eval.py generation --judge   also asks the LLM to judge if each answer is supported
  python eval.py all                  both
  python eval.py rescore              re-score the answers saved in eval_results/generation.jsonl
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
from llm import GENERAL_TAG, add_notes, ask_llm, build_messages, general_messages, source_tag
from store import Store, lecture_key

EVAL_SET = config.ROOT / "eval_set.jsonl"
OUT_DIR = config.ROOT / "eval_results"
KS = (1, 3, 5)                                       # cut-offs for Hit@k

ABSTAIN_PHRASES = ("not contain", "does not include", "doesn't include", "not mention",
                   "no information", "not covered", "not provide", "not in the sources",
                   "cannot answer", "can't answer", "cannot be answered", "unable to answer",
                   "do not cover", "does not cover", "don't cover", "doesn't cover", "not discussed",
                   "not specified", "no mention", "couldn't find", "could not find",
                   "can't find", "cannot find", "no relevant information")


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


def gold_items(q):
    return q["slides"] if "slides" in q else [q["extra"]]


def covered_slides(chunk, q):
    """Gold items of question q this chunk covers: slides it contains (slide chunk) or cites (guide
    chunk), or all gold items if it is the question's supplement topic. A question may have both
    "slides" and "extra" (slides cover it partly, the supplement fully): either one counts."""
    m = chunk["metadata"]
    if m["layer"] == "supplement":
        return set(gold_items(q)) if m.get("topic_id") == q.get("extra") else set()
    if "slides" not in q or chunk_lecture(chunk) != q["lecture"]:
        return set()
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
    out["recall"] = len(found) / len(gold_items(q))
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    return out


def print_score_gap(store, embedder, questions):
    """Best dense score per question, to choose config.MIN_TOP_SCORE (the relevance gate)."""
    scores = [(store.top_score(embedder.query(q["question"])), q["id"], no_answer_expected(q))
              for q in questions]
    ans = sorted(s for s in scores if not s[2])
    ref = sorted((s for s in scores if s[2]), reverse=True)
    if not ref:
        return
    print("\n  Top dense score (for MIN_TOP_SCORE; the gate refuses questions below it):")
    print("    lowest answerable:  " + "  ".join(f"{i} {v:.3f}" for v, i, _ in ans[:6]))
    print("    should be refused:  " + "  ".join(f"{i} {v:.3f}" for v, i, _ in ref))
    print(f"    current MIN_TOP_SCORE = {config.MIN_TOP_SCORE}")


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
               "gold": " or ".join(([f"{q['lecture']} {q['slides']}"] if "slides" in q else [])
                                   + ([f"Extra {q['extra']}"] if "extra" in q else []))}
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

    print_score_gap(store, embedder, questions)

    misses = [r for r in rows if r["context"]["hit@5"] == 0]
    if misses:
        print(f"\n  Gold slide not in the top 5 of the final context ({len(misses)}):")
        for r in misses:
            print(f"    {r['id']} gold {r['gold']:<22} got {' '.join(r['retrieved'][:5])}")
            print(f"        {r['question']}")
    print(f"\n  details: {save('retrieval', rows)}")


# ---------------- generation -------------------------------------------------

REFUSAL = "The lecture slides do not cover this."            # the exact sentence llm.py asks for
LOOP_RE = re.compile(r"(\S+)(?:\s+\1){29,}")                  # same token 30+ times in a row
LEC_RE = re.compile(r"\bL(?:ecture\s*)?0?(\d{1,2})\b\s*[:,]?", re.I)
ITEM_RE = re.compile(r"Guide\s+(\d+\.\d+)|Slides?\s+((?:\d+(?:\s*[–-]\s*\d+)?(?:\s*part\s*\d+)?"
                     r"(?:\s*(?:,|and|&)\s*(?:Slides?\s+)?)?)+)", re.I)


EXTRA_RE = re.compile(r"\bExtra\s+(X\d+)", re.I)


def parse_citations(answer):
    """All citations in an answer as ("L05", "slide", 12) / ("L05", "guide", "5.3").
    Handles every format seen so far: (L05 Slide 12), [L05: Slides 12–13], (L05: Slides 30, 31, 37),
    [L07 Guide 7.3, Slide 18], (L07: Slide 19, Slide 20), L06 Slide 28 part 2.
    A marker like "L2" in "L2 regularization" yields nothing, since no Slide/Guide follows it."""
    out = []
    marks = list(LEC_RE.finditer(answer))
    for i, m in enumerate(marks):
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(answer)
        seg = re.split(r"[)\]\n]", answer[m.end():stop], maxsplit=1)[0]
        lec = f"L{int(m.group(1)):02d}"
        for g in ITEM_RE.finditer(seg):
            if g.group(1):
                out.append((lec, "guide", g.group(1)))
                continue
            nums = re.sub(r"part\s*\d+", "", g.group(2), flags=re.I)
            for a, b in re.findall(r"(\d+)\s*[–-]\s*(\d+)", nums):
                if 0 <= int(b) - int(a) <= 30:
                    out += [(lec, "slide", n) for n in range(int(a), int(b) + 1)]
            nums = re.sub(r"\d+\s*[–-]\s*\d+", "", nums)
            out += [(lec, "slide", int(n)) for n in re.findall(r"\d+", nums)]
    out += [("EXTRA", "extra", t.upper()) for t in EXTRA_RE.findall(answer)]
    return list(dict.fromkeys(out))


def guide_slide_map(store):
    """(lecture, guide section) -> slides that section cites."""
    gm = {}
    for c in store.chunks:
        m = c["metadata"]
        if m["layer"] == "study_guide":
            key = (m["lecture_id"], m.get("section", "").split(" ")[0])
            gm.setdefault(key, set()).update(m.get("cited_slides", []))
    return gm


def citation_slides(cite, gm):
    lec, kind, num = cite
    if kind == "extra":
        return set()
    return {(lec, num)} if kind == "slide" else {(lec, n) for n in gm.get((lec, num), ())}


def is_refusal(answer):
    """Exact refusal sentence, or a short answer that is only a refusal. A long answer that merely
    notes something 'is not mentioned in the sources' still counts as answering."""
    a = norm(answer)
    return norm(REFUSAL) in a or (len(answer) < 600 and any(norm(p) in a for p in ABSTAIN_PHRASES))


def score_answer(q, answer, chunks, gm):
    """Score one answer against question q, given the chunks the model saw."""
    cites = parse_citations(answer)
    given = set()                                 # (lecture, slide) pairs the model had in front of it
    given_guides, given_extra = set(), set()
    for c in chunks:
        lec, m = chunk_lecture(c), c["metadata"]
        if m["layer"] == "supplement":
            given_extra.add(m.get("topic_id"))
        elif m["layer"] == "study_guide":
            given_guides.add((lec, m.get("section", "").split(" ")[0]))
            given.update((lec, n) for n in m.get("cited_slides", []))
        else:
            given.add((lec, m.get("slide")))
    valid = [(c[2] in given_extra) if c[1] == "extra" else ((c[0], c[2]) in given_guides) if c[1] == "guide"
             else ((c[0], c[2]) in given) for c in cites]
    uses_extra = any(c[1] == "extra" for c in cites)
    row = {"citations": [f"Extra {n}" if k == "extra" else f"{l} {k.capitalize()} {n}" for l, k, n in cites],
           "citations_valid": mean(valid) if cites else None,
           "abstained": is_refusal(answer),
           "general": GENERAL_TAG.lower() in answer.lower(),
           "looped": bool(LOOP_RE.search(answer)),
           "uses_extra": uses_extra,
           "labels_extra": norm("not from the lecture slides") in norm(answer) if uses_extra else None}
    if no_answer_expected(q):
        row["unanswerable"] = True
        row["kind"] = "not in slides" if q.get("in_slides") is False else "off-topic"
        return row
    a = norm(answer)
    hits = [f for f in q["facts"] if any(norm(alt) in a for alt in f.split("|"))]
    cites_gold = "extra" in q and any(c[1] == "extra" and c[2] == q["extra"] for c in cites)
    if "slides" in q:
        gold = {(q["lecture"], n) for n in q["slides"]}
        cites_gold = cites_gold or any(citation_slides(c, gm) & gold for c in cites)
    row.update({"facts_found": hits, "facts_missing": [f for f in q["facts"] if f not in hits],
                "fact_recall": len(hits) / len(q["facts"]), "cites_gold": float(cites_gold)})
    return row


JUDGE_PROMPT = """You check answers for a study assistant.
Below are SOURCES and an ANSWER. Decide whether every factual claim in the ANSWER is supported by the SOURCES.
Reply with exactly one word: SUPPORTED or UNSUPPORTED."""


def judge(answer, chunks):
    sources = "\n\n-----\n\n".join(f"{source_tag(c)}\n{c['text']}" for c in chunks)
    msgs = [{"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"SOURCES:\n\n{sources}\n\nANSWER:\n{answer}"}]
    verdict = ask_llm(msgs).strip().upper()
    return 0.0 if "UNSUPPORTED" in verdict else 1.0 if "SUPPORTED" in verdict else None


def eval_generation(store, embedder, questions, use_judge=False, saved=None):
    """saved: {id: previous row} to re-score old answers instead of generating new ones."""
    gm = guide_slide_map(store)
    config.LLM_TEMPERATURE = config.EVAL_TEMPERATURE    # deterministic: runs differ only by your changes
    rows = []
    for i, q in enumerate(questions, 1):
        qvec = embedder.query(q["question"])
        chunks = store.search(q["question"], qvec)
        if saved is not None:
            if q["id"] not in saved:
                continue
            answer, seconds = saved[q["id"]]["answer"], saved[q["id"]].get("seconds", 0.0)
        elif not store.relevant(qvec) and config.OFF_TOPIC == "refuse":   # relevance gate
            answer, seconds = REFUSAL, 0.0
        else:
            gated = not store.relevant(qvec)
            if gated:                                       # off-topic: no course sources
                chunks = []
            t0 = time.perf_counter()
            answer = ask_llm(general_messages(q["question"]) if gated
                             else build_messages(q["question"], chunks))
            seconds = round(time.perf_counter() - t0, 1)
            if gated:
                answer = GENERAL_TAG + "\n" + answer
        answer = add_notes(answer)
        row = {"id": q["id"], "area": q.get("area", "-"), "question": q["question"],
               "answer": answer, "seconds": seconds, **score_answer(q, answer, chunks, gm)}
        if use_judge and not row.get("unanswerable"):
            row["supported"] = judge(answer, chunks)
        rows.append(row)
        if saved is None:
            status = ("refused" if row["abstained"] else "general" if row.get("unanswerable") and row["general"]
                      else "UNTAGGED" if row.get("unanswerable") else f"facts {row['fact_recall']:.0%}")
            print(f"  [{i}/{len(questions)}] {q['id']} {status:<12} {seconds:5.1f}s  {q['question'][:60]}")
    report_generation(rows, use_judge)


def report_generation(rows, use_judge=False):
    ans = [r for r in rows if not r.get("unanswerable")]
    una = [r for r in rows if r.get("unanswerable")]
    with_cites = [r for r in rows if r["citations_valid"] is not None]
    print(f"\nGENERATION  ({len(ans)} answerable + {len(una)} unanswerable questions)")
    print(f"  fact recall (avg share of key facts in answer) {mean([r['fact_recall'] for r in ans]):6.2f}")
    print(f"  all facts present                              {mean([r['fact_recall'] == 1 for r in ans]):6.2f}")
    print(f"  cites a gold slide                             {mean([r['cites_gold'] for r in ans]):6.2f}")
    print(f"  citations point to given sources (valid)       {mean([r['citations_valid'] for r in with_cites]):6.2f}")
    print(f"  answers with no citation                       {mean([not r['citations'] for r in ans]):6.2f}")
    handled = [r["abstained"] or r.get("general", False) for r in una]
    print(f"  outside-source questions refused or tagged     {mean(handled):6.2f}"
          f"   (refused {sum(r['abstained'] for r in una)}, "
          f"tagged [General knowledge] {sum(r.get('general', False) and not r['abstained'] for r in una)})")
    for r, ok in zip(una, handled):
        if not ok:
            print(f"      {r['id']} ({r['kind']}) answered from memory WITHOUT the tag")
    print(f"  wrong refusal on answerable                    {mean([r['abstained'] for r in ans]):6.2f}")
    print(f"  answerable but tagged general knowledge        {mean([r.get('general', False) for r in ans]):6.2f}"
          "   (fine if only part of the answer)")
    print(f"  answers stuck in a repetition loop             {mean([r['looped'] for r in rows]):6.2f}")
    ext = [r for r in rows if r.get("uses_extra")]
    if ext:
        print(f"  supplement answers labelled 'not from slides'  {mean([r['labels_extra'] for r in ext]):6.2f}"
              f"   ({len(ext)} answers used [Extra] notes)")
    if use_judge:
        judged = [r["supported"] for r in ans if r.get("supported") is not None]
        print(f"  judged supported by sources (LLM judge)        {mean(judged):6.2f}   ({len(judged)} judged)")
    print(f"  avg seconds per answer                         {mean([r['seconds'] for r in rows]):6.1f}")
    areas = by_area(ans, "fact_recall")
    if len(areas) > 1:
        print("\n  fact recall by area:")
        for a, (v, n) in areas.items():
            print(f"    {a:<20} {v:5.2f}   ({n} questions)")

    weak = [r for r in ans if r["fact_recall"] < 1 or not r["cites_gold"] or r["abstained"] or r["looped"]
            or r.get("general")]
    if weak:
        print(f"\n  Answers to check ({len(weak)}):")
        for r in weak:
            why = []
            if r["looped"]:
                why.append("REPETITION LOOP")
            if r["facts_missing"]:
                why.append("missing " + ", ".join(repr(f.split('|')[0]) for f in r["facts_missing"]))
            if not r["citations"]:
                why.append("no citation")
            elif not r["cites_gold"]:
                why.append("cites only non-gold slides")
            if r["abstained"]:
                why.append("refused")
            if r.get("general"):
                why.append("partly general knowledge")
            print(f"    {r['id']}: {'; '.join(why)}")
    print(f"\n  details (full answers): {save('generation', rows)}")


# ---------------- main -------------------------------------------------------

def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd not in ("retrieval", "generation", "all", "rescore"):
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
    if cmd == "rescore":                # re-score saved answers with the current scoring code
        path = OUT_DIR / "generation.jsonl"
        saved = {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}
        print(f"Re-scoring {len(saved)} saved answers from {path} (no LLM calls)")
        eval_generation(store, embedder, questions, saved=saved)


if __name__ == "__main__":
    main(sys.argv)