"""The corpus, its validation, and what retrieval actually recovers.

Runs with nothing installed. The vector store and the answer chain need Ollama
and are not exercised here; `python3 -m natech.cli` is the interactive system.

Run: python3 demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from natech import corpus, groundedness, retrieval

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

    heading(3, "Extraction artefacts, which do not block but do cost recall")
    defects = corpus.text_defects(sections)
    print("  {0} found across {1} sections.".format(len(defects), len(sections)))
    print()
    for finding in defects:
        print("  " + str(finding))
    print()
    print("  Three, in a corpus of 107,312 characters. Small enough to state at")
    print("  its real size rather than dress up. The third is the one with a")
    print("  cost: ISO-310002 is ISO-31000 with a superscript footnote welded")
    print("  on by the PDF-to-text step, and it embeds as a different token, so")
    print("  a question about ISO 31000 will not retrieve chapter 3.")
    print()
    print("  These are review, not blocking. They degrade retrieval quietly,")
    print("  which is harder to notice than a refusal, which is the argument")
    print("  for looking for them at all.")

    heading(4, "Retrieval, measured rather than read")
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

    heading(5, "Where the baseline fails, and why that is informative")
    at_two = retrieval.evaluate(retrieve, k=2)
    for result in at_two["results"]:
        print("  " + str(result))
    print()
    print('  "What is a Natech accident?" is never retrieved at any k. Every')
    print("  content word in it is either stopped or appears in most sections,")
    print("  so term overlap has nothing to work with. That is exactly the")
    print("  question an embedding retriever should win, and the comparison")
    print("  only exists because the baseline was built.")

    heading(6, "The failure a recall number cannot see")
    print("Retrieval can work and the answer can still be wrong. Section 4")
    print("measures whether the right passage came back. It says nothing about")
    print("whether the generated text is supported by the passage that did.")
    print()
    by_chapter = {s.chapter: s for s in sections}
    evidence = [groundedness.Evidence(s.chapter, s.title, s.content)
                for s in (by_chapter["4.3"], by_chapter["4.6.1.1"])]
    print("  Evidence, real text from the corpus:")
    for item in evidence:
        print("    Chapter {0:<10} {1}".format(item.chapter, item.title[:56]))
    print()

    trials = [
        ("taken from the evidence",
         "Lightning strikes have caused over 90% of all tank fires.", None),
        ("the same claim with the number changed",
         "Lightning strikes have caused over 75% of all tank fires.", None),
        ("true, but cited to the wrong chapter",
         "Damage of 10 mm is treated as minor severity.", ["4.3"]),
        ("prose with nothing checkable in it",
         "Operators should adopt a precautionary approach throughout.", None),
    ]
    for label, text, cited in trials:
        report = groundedness.check(text, evidence, cited=cited)
        print("  {0:<40} {1}".format(label, report.verdict))
        for problem in report.problems():
            for line in str(problem).split("\n")[1:]:
                print("  " + line.strip())
    print()
    print("  The second is the one worth the whole module. Retrieval succeeded,")
    print("  the cited chapter is right, and the number is invented. No recall")
    print("  or precision figure moves.")
    print()
    print("  The fourth is the honest half. That claim is not endorsed, it is")
    print("  declined: there is nothing in it this method can check. A screen")
    print("  that returned a pass there would be converting an unread answer")
    print("  into a checked one, which is worse than having no screen at all.")
    print()
    print("  Across the whole corpus only 22 tokens are checkable at all, in")
    print("  107,312 characters. This is prose guidance, not a table of")
    print("  thresholds, so the screen has little to work with here and says so")
    print("  through checkable_share rather than reporting a quiet run as clean.")


if __name__ == "__main__":
    main()
