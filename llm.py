"""Prompt building and the call to llama-server."""
import requests

import config

SYSTEM = """You are a study assistant for a university NLP course.
Answer the question using ONLY the sources below. Each source starts with a tag like [L03 Slide 12] or [L03 Guide 3.5].
Rules:
- Cite the tag(s) you used after each claim, e.g. "... (L03 Slide 12)".
- If the sources do not contain the answer, say so plainly instead of guessing.
- Prefer the lecturer's wording and notation. Keep formulas exactly as given.
- Answer in the same language as the question."""


def source_tag(chunk):
    m = chunk["metadata"]
    lec = m.get("lecture_id") or m.get("source_file", "")[:9].replace("Lecture", "L")
    if m["layer"] == "study_guide":
        sec = m.get("section", "").split(" ")[0]
        return f"[{lec} Guide {sec}]"
    part = f" part {m['part']}" if m.get("parts", 1) > 1 else ""
    return f"[{lec} Slide {m.get('slide')}{part}]"


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
        "stream": stream,
    })
    r.raise_for_status()
    if not stream:
        return r.json()["choices"][0]["message"]["content"].strip()
    return _stream(r)


def _stream(r):
    import json
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        delta = json.loads(data)["choices"][0]["delta"]
        if delta.get("content"):          # first chunk carries only the role, content=null
            yield delta["content"]