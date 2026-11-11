"""Prompt building and the call to llama-server."""
import re

import requests

import config

SYSTEM = """You are a study assistant for a university NLP course.
Answer using the sources below. Each source starts with a tag such as [Lxx Slide n], [Lxx Guide x.y]
or [Extra Xn]. Cite only tags that appear in the sources below.
Sources tagged [Extra Xn] are supplementary textbook notes, not from the lecture slides.
Rules:
1. The sources teach concepts. You MAY apply them to new examples and do calculations
   (e.g. classify a new sentence pair, compute a vector). Show each step, but keep it short:
   never write out long vectors or lists element by element. Give final numbers as decimals
   (e.g. 0.67, not only a formula).
2. If the TOPIC of the question is not covered by the sources at all, start your reply with
   the line "[General knowledge]" and then answer briefly from your own knowledge, without any
   source tags. Never stretch a source about a different topic to make it fit.
3. If the sources cover only part of the question, answer that part with source tags first.
   Then write the line "[General knowledge]" and answer the rest from your own knowledge.
   Never put a source tag on anything written from your own knowledge.
4. After each claim, cite the tag of the source it came from, in brackets.
5. Prefer the lecture sources over [Extra] sources whenever they answer the question.
6. Prefer the lecturer's wording and notation. Keep formulas exactly as given.
7. Finish with one line: "Sources:" followed by the tags you used.
8. Answer in the same language as the question."""

GENERAL_SYSTEM = """You are a study assistant for a university NLP course. The course materials do not
cover the question below, so answer it briefly and accurately from your own knowledge.
If you are not sure, say so. Do not cite any sources."""

REFUSAL = "The lecture slides do not cover this."
GENERAL_TAG = "[General knowledge]"
EXTRA_NOTE = ("\n\nNote: the parts that cite Extra notes come from supplementary textbook material, "
              "not from the lecture slides.")
GENERAL_NOTE = ("\n\nNote: the parts marked [General knowledge] are not from the lecture slides or the "
                "study notes. The model wrote them from memory, so they may be wrong: check them.")


def source_tag(chunk):
    m = chunk["metadata"]
    lec = m.get("lecture_id") or m.get("source_file", "")[:9].replace("Lecture", "L")
    if m["layer"] == "supplement":
        return f"[Extra {m['topic_id']}]"
    if m["layer"] == "study_guide":
        sec = m.get("section", "").split(" ")[0]
        return f"[{lec} Guide {sec}]"
    part = f" part {m['part']}" if m.get("parts", 1) > 1 else ""
    return f"[{lec} Slide {m.get('slide')}{part}]"


def add_notes(answer):
    """Label non-lecture material in code; a small model is unreliable at doing it itself."""
    if re.search(r"\bExtra\s+X\d", answer) and EXTRA_NOTE.strip() not in answer:
        answer += EXTRA_NOTE
    if GENERAL_TAG.lower() in answer.lower() and GENERAL_NOTE.strip() not in answer:
        answer += GENERAL_NOTE
    return answer


def general_messages(question):
    """For questions the relevance gate finds off-topic: no sources, answer from general knowledge."""
    return [{"role": "system", "content": GENERAL_SYSTEM}, {"role": "user", "content": question}]


def build_messages(question, chunks):
    blocks = [f"{source_tag(c)}\n{c['text']}" for c in chunks]
    user = "SOURCES:\n\n" + "\n\n-----\n\n".join(blocks) + f"\n\nQUESTION: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def ask_llm(messages, stream=False):
    r = requests.post(config.LLM_URL, timeout=config.LLM_TIMEOUT, stream=stream, json={
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        "repeat_penalty": config.LLM_REPEAT_PENALTY,   # llama-server option: stops "0 0 0 ..." loops
        "stream": stream,
        **config.LLM_EXTRA,                            # e.g. turn off Qwen "thinking"
    })
    r.raise_for_status()
    if not stream:
        msg = r.json()["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        return content or (NO_ANSWER if msg.get("reasoning_content") else "")
    return _stream(r)


NO_ANSWER = ("[No answer: the model spent all its tokens thinking. Turn thinking off "
             "(see LLM_EXTRA in config.py) or raise LLM_MAX_TOKENS.]")


def _stream(r):
    import json
    got_answer = thinking = False
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        choices = json.loads(data).get("choices") or [{}]
        delta = choices[0].get("delta", {})
        if delta.get("reasoning_content") and not thinking:
            thinking = True
            print("(model is thinking...)", flush=True)
        if delta.get("content"):          # first chunk carries only the role, content=null
            got_answer = True
            yield delta["content"]
    if not got_answer:
        yield NO_ANSWER if thinking else "[No answer: the server returned empty text.]"