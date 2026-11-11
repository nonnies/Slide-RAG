"""Slide RAG: ask questions about your lecture slides with a local LLM.

  python rag.py index                 build the index (run once, and after chunks change)
  python rag.py ask "question"        answer one question
  python rag.py chat                  interactive loop
  python rag.py eval questions.txt    show what gets retrieved for each question (no LLM)
"""
import sys

import config
from embed import Embedder
from llm import GENERAL_TAG, REFUSAL, add_notes, ask_llm, build_messages, general_messages, source_tag
from store import Store, build_index


def retrieve(store, embedder, question):
    return store.search(question, embedder.query(question))


def answer(store, embedder, question, show_sources=True):
    qvec = embedder.query(question)
    text = ""
    if store.relevant(qvec):
        chunks = store.search(question, qvec)
        messages = build_messages(question, chunks)
        if show_sources:
            print("Sources: " + "  ".join(source_tag(c) for c in chunks), flush=True)
    elif config.OFF_TOPIC == "refuse":                 # relevance gate (config.MIN_TOP_SCORE)
        print(REFUSAL)
        return
    else:                                              # off-topic: general knowledge, tagged by code
        messages = general_messages(question)
        text = GENERAL_TAG + "\n"
        print(text, end="", flush=True)
    for piece in ask_llm(messages, stream=True):
        print(piece, end="", flush=True)
        text += piece
    print(add_notes(text)[len(text):])                 # labels for Extra / general-knowledge parts


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "help"

    if cmd == "index":
        build_index(Embedder())

    elif cmd == "ask" and len(argv) > 2:
        answer(Store(), Embedder(), " ".join(argv[2:]))

    elif cmd == "chat":
        store, embedder = Store(), Embedder()
        print("Ask about the lectures (empty line to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            answer(store, embedder, q)

    elif cmd == "eval" and len(argv) > 2:
        store, embedder = Store(), Embedder()
        for q in open(argv[2], encoding="utf-8"):
            q = q.strip()
            if not q or q.startswith("#"):
                continue
            print(f"\nQ: {q}")
            for c in retrieve(store, embedder, q):
                first = c["text"].split("\n\n", 1)[-1].replace("\n", " ")[:90]
                print(f"   {source_tag(c):<22} {first}")

    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)