"""Measuring retrieval, and the baseline it has to beat.

Every test here runs against a stub or against the keyword baseline, so the
whole evaluation is checkable without Ollama. That is the reason `evaluate`
takes a retrieval function rather than a store.
"""

import os
import unittest

from natech import corpus, retrieval

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(HERE, "data", "guidance.csv")

PROBES = (
    retrieval.Probe("first question", ("A",)),
    retrieval.Probe("second question", ("B",)),
    retrieval.Probe("third question", ("C", "D"), "two acceptable chapters"),
)


def fixed(order):
    """A retriever that always returns the same chapters, for arithmetic tests."""
    def retrieve(question, k):
        return list(order)[:k]
    return retrieve


# Built once and held at module level. Assigning a plain function to a class
# attribute makes Python bind it as a method, so self arrives as the question.
_CACHE = {}


def baseline():
    if "retrieve" not in _CACHE:
        sections = corpus.load(REAL)
        _CACHE["sections"] = sections
        _CACHE["retrieve"] = retrieval.keyword_retriever(sections)
    return _CACHE["retrieve"]


def sections():
    baseline()
    return _CACHE["sections"]


class TestEvaluate(unittest.TestCase):
    def test_everything_found_at_rank_one_scores_one(self):
        def perfect(question, k):
            return {"first question": ["A"], "second question": ["B"],
                    "third question": ["C"]}[question][:k]
        result = retrieval.evaluate(perfect, PROBES, k=1)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["mrr"], 1.0)

    def test_nothing_found_scores_zero_and_names_what_was_missed(self):
        result = retrieval.evaluate(fixed(["Z"]), PROBES, k=1)
        self.assertEqual(result["recall_at_k"], 0.0)
        self.assertEqual(result["mrr"], 0.0)
        self.assertEqual(len(result["missed"]), 3)

    def test_rank_two_is_worth_half_as_much_in_the_mrr(self):
        def second(question, k):
            return ["Z", {"first question": "A", "second question": "B",
                          "third question": "C"}[question]][:k]
        at_one = retrieval.evaluate(second, PROBES, k=1)
        at_two = retrieval.evaluate(second, PROBES, k=2)
        self.assertEqual(at_one["recall_at_k"], 0.0)
        self.assertEqual(at_two["recall_at_k"], 1.0)
        self.assertAlmostEqual(at_two["mrr"], 0.5, places=4)

    def test_either_acceptable_chapter_counts_as_a_hit(self):
        """Some questions are genuinely answered by more than one section, and
        forcing a single gold chapter scores a correct retrieval as wrong."""
        for chapter in ("C", "D"):
            result = retrieval.evaluate(fixed([chapter]), (PROBES[2],), k=1)
            self.assertEqual(result["recall_at_k"], 1.0, chapter)

    def test_raising_k_never_lowers_recall(self):
        recalls = [r["recall_at_k"] for r in retrieval.sweep(baseline())]
        for earlier, later in zip(recalls, recalls[1:]):
            self.assertGreaterEqual(later, earlier)

    def test_an_empty_probe_set_reports_none_rather_than_zero(self):
        result = retrieval.evaluate(fixed(["A"]), (), k=1)
        self.assertEqual(result["probes"], 0)
        self.assertIsNone(result["recall_at_k"])


class TestKeywordBaseline(unittest.TestCase):
    def test_it_returns_at_most_k_chapters(self):
        for k in (1, 2, 5):
            self.assertLessEqual(len(baseline()("equipment damage", k)), k)

    def test_it_returns_chapters_that_exist(self):
        known = {s.identifier for s in sections()}
        for chapter in baseline()("emergency planning and response", 5):
            self.assertIn(chapter, known)

    def test_a_question_sharing_no_terms_returns_nothing(self):
        """Better than returning the least-bad chapter. A retriever that always
        returns something makes an unanswerable question look answerable."""
        self.assertEqual(baseline()("zzzz qqqq", 3), [])

    def test_it_is_deterministic(self):
        self.assertEqual(baseline()("safety barriers and utilities", 3),
                         baseline()("safety barriers and utilities", 3))

    def test_the_baseline_finds_something_on_most_probes(self):
        """It is a baseline, not a strawman. If it found nothing, beating it
        would prove nothing about the embedding retriever."""
        self.assertGreater(retrieval.evaluate(baseline(), k=8)["recall_at_k"], 0.5)


class TestTheFindingTheSweepProduces(unittest.TestCase):
    """The measurement the April system could not make.

    k was 2 and there was no way to ask whether that was enough. These figures
    are quoted in the README, so they are pinned here.
    """

    @classmethod
    def setUpClass(cls):
        cls.swept = retrieval.sweep(baseline())
        cls.by_k = {r["k"]: r for r in cls.swept}

    def test_recall_at_the_original_k_of_two(self):
        self.assertAlmostEqual(self.by_k[2]["recall_at_k"], 4 / 12.0, places=4)

    def test_recall_at_three_is_markedly_better(self):
        self.assertAlmostEqual(self.by_k[3]["recall_at_k"], 7 / 12.0, places=4)

    def test_the_step_from_two_to_three_is_the_argument(self):
        gain = self.by_k[3]["recall_at_k"] - self.by_k[2]["recall_at_k"]
        self.assertGreater(gain, 0.2)

    def test_the_report_does_not_claim_a_plateau_it_has_not_found(self):
        """An earlier version printed the largest k tried as the plateau, which
        reports the edge of the sweep as a finding about retrieval."""
        text = retrieval.report(self.swept)
        self.assertIn("still improving", text)
        self.assertNotIn("Recall stops improving", text)

    def test_a_sweep_that_does_plateau_says_so(self):
        def plateaus(question, k):
            return ["A"][:k]
        probes = (retrieval.Probe("q", ("A",)),)
        text = retrieval.report(retrieval.sweep(plateaus, probes, ks=(1, 2, 3)))
        self.assertIn("Recall stops improving at k = 1", text)


class TestLangchainAdapter(unittest.TestCase):
    def test_it_reads_the_chapter_off_the_metadata(self):
        """Which is the reason the metadata is carried at all."""
        class Document:
            def __init__(self, chapter):
                self.metadata = {"chapter": chapter, "title": "t"}

        class Retriever:
            def invoke(self, question, k=2):
                return [Document("4.2"), Document("5.1")]

        adapted = retrieval.from_langchain(Retriever())
        self.assertEqual(adapted("anything", 2), ["4.2", "5.1"])

    def test_it_copes_with_a_retriever_that_takes_no_k(self):
        class Document:
            def __init__(self, chapter):
                self.metadata = {"chapter": chapter}

        class OldRetriever:
            def invoke(self, question):
                return [Document("1"), Document("2"), Document("3")]

        adapted = retrieval.from_langchain(OldRetriever())
        self.assertEqual(adapted("anything", 2), ["1", "2"])


if __name__ == "__main__":
    unittest.main()
