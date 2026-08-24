"""The interactive loop, as it was in April, with the citation added.

Run: python3 -m natech.cli
Needs Ollama running. For everything that does not, see demo.py.
"""

import argparse
import sys

from .answer import DEFAULT_MODEL, answer
from .store import BackendMissing, DEFAULT_K, build, retriever


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default="data/guidance.csv")
    parser.add_argument("--store", default="./chrome_langchain_db")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="passages retrieved per question. The April value "
                             "is 2; retrieval.py measures what raising it buys")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the index even if it looks complete")
    args = parser.parse_args(argv)

    try:
        store = build(corpus_path=args.corpus, store_path=args.store,
                      rebuild=args.rebuild)
        search = retriever(store, k=args.k)
    except BackendMissing as error:
        print(error)
        return 1

    while True:
        print("\n\n------------------------------------")
        question = input("Ask your question (q to quit): ")
        print("\n \n ")
        if question.strip().lower() == "q":
            break
        if not question.strip():
            continue
        print(answer(question, search, model_name=args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
