"""Every number the demo and the README state, checked against the corpus.

Figures written into prose go stale silently. A corpus re-extraction that
changes the character count or the defect count leaves the old number sitting
in the text looking authoritative, and nobody re-reads a paragraph they wrote
months ago. So each one is asserted here against the file it came from, and the
test names say where the claim appears.
"""

import io
import os
import re
import unittest

from natech import corpus, groundedness as g

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(HERE, "data", "guidance.csv")
DEMO = os.path.join(HERE, "demo.py")
README = os.path.join(HERE, "README.md")


def read(path):
    with io.open(path, encoding="utf-8") as handle:
        return handle.read()


class TestCorpusFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sections = corpus.load(REAL)
        cls.stats = corpus.statistics(cls.sections)

    def test_the_character_count_quoted_in_the_demo_is_the_real_one(self):
        quoted = "{0:,}".format(self.stats["characters"])
        self.assertIn(quoted, read(DEMO),
                      "demo.py quotes a character count the corpus no longer has")

    def test_the_section_count_is_forty_four(self):
        self.assertEqual(self.stats["sections"], 44)

    def test_the_corpus_still_loads_without_blocking_findings(self):
        blocking = [f for f in corpus.check(self.sections)
                    if f.severity == "blocking"]
        self.assertEqual(blocking, [])


class TestDefectFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defects = corpus.text_defects(corpus.load(REAL))

    def test_there_are_three_and_the_demo_says_three(self):
        self.assertEqual(len(self.defects), 3)
        self.assertIn("Three, in a corpus", read(DEMO))

    def test_each_one_is_review_severity(self):
        for finding in self.defects:
            self.assertEqual(finding.severity, "review")

    def test_the_iso_defect_is_still_the_one_described(self):
        """The demo names ISO-310002 specifically. If a re-extraction fixes it,
        that paragraph has to go."""
        self.assertTrue(any("ISO" in f.problem for f in self.defects))


class TestCheckableTokenFigure(unittest.TestCase):
    def test_the_token_count_quoted_in_the_demo_is_the_real_one(self):
        """The demo says only 22 tokens in the whole corpus are checkable.
        That number is the argument for reporting checkable_share at all, so it
        has to be true.

        It was written as 23 first, from a probe run taken before "approach"
        was added to the structural words. This test caught it.
        """
        sections = corpus.load(REAL)
        total = sum(len(g.tokens(s.content)) for s in sections)
        self.assertEqual(total, 22)
        self.assertIn("only 22 tokens", read(DEMO))


class TestTheDemoVerdicts(unittest.TestCase):
    """The four trials in section 6 must actually produce four verdicts.

    Without this, a change to the extractor could collapse them all to
    UNVERIFIABLE and the demo would still print, still look confident, and
    demonstrate nothing.
    """

    @classmethod
    def setUpClass(cls):
        sections = {s.chapter: s for s in corpus.load(REAL)}
        cls.evidence = [g.Evidence(s.chapter, s.title, s.content)
                        for s in (sections["4.3"], sections["4.6.1.1"])]

    def verdict(self, text, cited=None):
        return g.check(text, self.evidence, cited=cited).verdict

    def test_the_quantity_from_the_corpus_is_not_contradicted(self):
        self.assertEqual(
            self.verdict("Lightning strikes have caused over 90% of all tank fires."),
            g.NOT_CONTRADICTED)

    def test_changing_that_number_makes_it_unsupported(self):
        self.assertEqual(
            self.verdict("Lightning strikes have caused over 75% of all tank fires."),
            g.UNSUPPORTED)

    def test_citing_the_wrong_chapter_is_misattributed(self):
        self.assertEqual(
            self.verdict("Damage of 10 mm is treated as minor severity.",
                         cited=["4.3"]),
            g.MISATTRIBUTED)

    def test_unquantified_prose_is_unverifiable(self):
        self.assertEqual(
            self.verdict("Operators should adopt a precautionary approach."),
            g.UNVERIFIABLE)

    def test_all_four_are_distinct(self):
        verdicts = {
            self.verdict("Lightning strikes have caused over 90% of all tank fires."),
            self.verdict("Lightning strikes have caused over 75% of all tank fires."),
            self.verdict("Damage of 10 mm is treated as minor severity.", ["4.3"]),
            self.verdict("Operators should adopt a precautionary approach."),
        }
        self.assertEqual(len(verdicts), 4)


class TestReadme(unittest.TestCase):
    def test_every_python_file_listed_in_the_readme_exists(self):
        listed = re.findall(r"`([A-Za-z_][A-Za-z0-9_/]*\.py)`", read(README))
        for name in sorted(set(listed)):
            self.assertTrue(os.path.exists(os.path.join(HERE, name)),
                            "README lists {0}, which is not in the repo".format(name))

    def test_the_test_count_in_the_readme_matches_the_suite(self):
        import unittest as ut
        total = ut.defaultTestLoader.discover(
            os.path.join(HERE, "tests"), top_level_dir=HERE).countTestCases()
        self.assertIn("{0} tests".format(total), read(README),
                      "README states a test count the suite does not have")


if __name__ == "__main__":
    unittest.main()
