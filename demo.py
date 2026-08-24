"""The corpus, its validation, and what retrieval actually recovers.

Runs with nothing installed. The vector store and the answer chain need Ollama
and are not exercised here; `python3 -m natech.cli` is the interactive system.

Run: python3 demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from natech import corpus, retrieval

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "data", "guidance.csv")


def heading(number, text):
    print()
    print("=" * 76)
    print("{0}. {1}".format(number, text))
    print("=" * 76)


def main():
    heading(1, "The corpus")
    sections = corpus.load(CORPUS)
    print(corpus.report(sections, corpus.check(sections)))
    print()
    print("  Segmented at chapter granularity, structure kept rather than")
    print("  flattened, so chapter and title travel with the text into the")
    print("  index and an answer can name the section it came from.")
    print()
    for section in sections[:4]:
        print("    {0:>7}  {1:<52}{2:>6} chars".format(
            section.chapter, section.title[:50], section.characters))
    print("    {0:>7}  {1}".format("...", ""))
    for section in sections[-2:]:
        print("    {0:>7}  {1:<52}{2:>6} chars".format(
            section.chapter, section.title[:50], section.characters))

    heading(2, "What the validation catches, and why it blocks")
    broken = [
        corpus.Section("6", "A section with no body", ""),
        corpus.Section("6", "A duplicate chapter number", "x" * 500),
        corpus.Section("", "No chapter number at all", "x" * 500),
        corpus.Section("7", "A heading captured without its body", "See section 4."),
    ]
    for finding in corpus.check(broken):
        print("  " + str(finding))
    print()
    print("  The first is the one that matters. An empty section embeds to")
    print("  something meaningless, sits in the index, and comes back as")
    print("  context for a question it says nothing about. The April loader")
    print("  took it without complaint, because nothing was looking.")

    heading(3, "Retrieval, measured rather than read")
    print("The April system retrieved two passages per question and quality was")
    print("judged by reading them. That cannot answer whether two is enough.")
    print()
    retrieve = retrieval.keyword_retriever(sections)
    results = retrieval.sweep(retrieve)
    print(retrieval.report(results))
    print()
    print("  These are the term-overlap baseline, not the embedding retriever,")
    print("  which needs Ollama. The baseline is here so the embedding version")
    print("  has something to beat: a dense retriever that cannot beat counting")
    print("  shared words is not paying for the model server it requires.")
    print()
    print("  The number that matters is the step from k=2 to k=3. The original")
    print("  k of 2 was a guess, and on this probe set it costs a third of the")
    print("  questions the corpus can actually answer.")

    heading(4, "Where the baseline fails, and why that is informative")
    at_two = retrieval.evaluate(retrieve, k=2)
    for result in at_two["results"]:
        print("  " + str(result))
    print()
    print('  "What is a Natech accident?" is never retrieved at any k. Every')
    print("  content word in it is either stopped or appears in most sections,")
    print("  so term overlap has nothing to work with. That is exactly the")
    print("  question an embedding retriever should win, and the comparison")
    print("  only exists because the baseline was built.")


if __name__ == "__main__":
    main()
