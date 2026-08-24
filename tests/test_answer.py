"""The Answer object and the audit hanging off it.

No Ollama and no Chroma here. `answer()` needs both, but everything worth
testing about what it returns does not: the evidence it carries, the citations
it derives, and whether the audit can find a fabricated quantity. Those are
exercised by constructing the Answer directly, which is also the reason
generation is a thin function over a dataclass rather than a class that does
the retrieving, the generating and the reporting at once.
"""

import unittest

from natech.answer import Answer
from natech.groundedness import (Evidence, MISATTRIBUTED, SUPPORTED,
                                 UNSUPPORTED, UNVERIFIABLE)

EVIDENCE = (
    Evidence("3", "Siting", "Facilities shall maintain a minimum separation "
                            "of 500 m from the site boundary."),
    Evidence("7", "Damage", "Overpressure above 10 kPa is treated as causing "
                            "structural damage to ordinary buildings."),
)


def make(text, evidence=EVIDENCE, question="What separation is required?"):
    return Answer(question=question, text=text, evidence=evidence,
                  model="test")


class TestAnswer(unittest.TestCase):
    def test_sections_are_derived_from_the_evidence(self):
        """One source of truth. Storing both invites them to disagree."""
        self.assertEqual(make("x").sections, (("3", "Siting"), ("7", "Damage")))

    def test_citations_name_chapter_and_title(self):
        self.assertEqual(make("x").citations(),
                         ["Chapter 3, Siting", "Chapter 7, Damage"])

    def test_an_answer_with_no_evidence_says_so(self):
        self.assertIn("no sections retrieved", str(make("x", evidence=())))

    def test_the_text_and_the_sources_both_appear(self):
        rendered = str(make("A separation of 500 m applies."))
        self.assertIn("500 m", rendered)
        self.assertIn("Chapter 3", rendered)


class TestAudit(unittest.TestCase):
    def test_a_quantity_taken_from_the_evidence_passes(self):
        self.assertEqual(make("A separation of 500 m is required.").audit().verdict,
                         SUPPORTED)

    def test_a_fabricated_quantity_is_caught(self):
        """The failure retrieval metrics cannot see: retrieval worked, and the
        model still produced a number that is not in what it retrieved."""
        report = make("A separation of 800 m is required.").audit()
        self.assertEqual(report.verdict, UNSUPPORTED)
        self.assertIn("800 m", str(report))

    def test_citing_the_wrong_chapter_for_a_true_claim_is_caught(self):
        report = make("Overpressure above 10 kPa causes damage.").audit(cited=["3"])
        self.assertEqual(report.verdict, MISATTRIBUTED)

    def test_the_audit_carries_the_question(self):
        report = make("A separation of 500 m is required.").audit()
        self.assertEqual(report.question, "What separation is required?")

    def test_an_answer_with_no_evidence_cannot_support_anything(self):
        report = make("A separation of 500 m is required.", evidence=()).audit()
        self.assertEqual(report.verdict, UNSUPPORTED)

    def test_prose_with_no_quantities_gets_no_opinion(self):
        report = make("Operators should adopt a precautionary approach.").audit()
        self.assertEqual(report.verdict, UNVERIFIABLE)
        self.assertEqual(report.checkable_share, 0.0)

    def test_the_audit_counts_every_claim_exactly_once(self):
        report = make("A separation of 500 m is required. A separation of "
                      "800 m is not. Operators should be careful.").audit()
        self.assertEqual(len(report.claims), 3)
        self.assertEqual(sum(report.counts.values()), 3)


class TestEvidenceIsActuallyKept(unittest.TestCase):
    """The change this file exists for.

    Before it, Answer held (chapter, title) pairs and nothing else, so there
    was no way to check an answer against its sources without going back to the
    retriever by hand. The audit is only possible because the passage text
    survives generation.
    """

    def test_the_passage_text_survives_on_the_answer(self):
        self.assertIn("500 m", make("x").evidence[0].content)

    def test_an_audit_is_impossible_without_it(self):
        stripped = tuple(Evidence(e.chapter, e.title, "") for e in EVIDENCE)
        report = make("A separation of 500 m is required.",
                      evidence=stripped).audit()
        self.assertEqual(report.verdict, UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
