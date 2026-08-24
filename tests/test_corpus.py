"""Loading the guidance, and refusing to index it when it is broken.

The refusal tests are the ones that matter. A retrieval system built on a corpus
with holes in it answers confidently from the sections that survived, and the
sections that did not are invisible from outside.
"""

import io
import os
import shutil
import tempfile
import unittest

from natech import corpus

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(HERE, "data", "guidance.csv")

HEADER = "Chapter,Title,Content\n"
BODY = "x" * 200


def write(rows, directory):
    path = os.path.join(directory, "corpus.csv")
    with io.open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(HEADER)
        for chapter, title, content in rows:
            handle.write('{0},{1},"{2}"\n'.format(chapter, title, content))
    return path


class TestReading(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="natech_")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_a_missing_file_raises_rather_than_returning_nothing(self):
        with self.assertRaises(corpus.CorpusError):
            corpus.read(os.path.join(self.directory, "absent.csv"))

    def test_a_missing_column_is_named_in_the_error(self):
        path = os.path.join(self.directory, "wrong.csv")
        with io.open(path, "w", encoding="utf-8") as handle:
            handle.write("Chapter,Body\n1,text\n")
        with self.assertRaises(corpus.CorpusError) as caught:
            corpus.read(path)
        self.assertIn("Title", str(caught.exception))
        self.assertIn("Content", str(caught.exception))

    def test_whitespace_around_fields_is_stripped(self):
        path = write([("  1 ", " Introduction ", BODY)], self.directory)
        section = corpus.read(path)[0]
        self.assertEqual(section.chapter, "1")
        self.assertEqual(section.title, "Introduction")

    def test_an_empty_corpus_is_refused_by_load(self):
        path = write([], self.directory)
        with self.assertRaises(corpus.CorpusError):
            corpus.load(path)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="natech_")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_an_empty_section_blocks_the_load(self):
        """The defect this module exists for. An empty Content field became a
        Document with empty page_content, which embeds to something meaningless,
        sits in the store, and is retrievable as context for a question it says
        nothing about. Nothing raised and nothing logged."""
        path = write([("1", "Introduction", BODY), ("2", "Empty", "")], self.directory)
        with self.assertRaises(corpus.CorpusError) as caught:
            corpus.load(path)
        self.assertIn("empty content", str(caught.exception))

    def test_a_duplicate_chapter_blocks_the_load(self):
        """Documents are keyed on the chapter, so a duplicate silently
        overwrites rather than adding."""
        path = write([("1", "One", BODY), ("1", "Also one", BODY)], self.directory)
        with self.assertRaises(corpus.CorpusError) as caught:
            corpus.load(path)
        self.assertIn("duplicate", str(caught.exception))

    def test_a_missing_chapter_number_blocks_the_load(self):
        path = write([("", "No number", BODY)], self.directory)
        with self.assertRaises(corpus.CorpusError):
            corpus.load(path)

    def test_a_very_short_section_is_flagged_for_review_not_blocked(self):
        """Usually a heading captured without its body. Worth a human look, and
        blocking on it would stop a corpus that is mostly fine."""
        path = write([("1", "Introduction", BODY), ("2", "Stub", "short")], self.directory)
        sections = corpus.read(path)
        findings = corpus.check(sections)
        short = [f for f in findings if "below the" in f.problem]
        self.assertEqual(len(short), 1)
        self.assertEqual(short[0].severity, "review")
        corpus.load(path)  # must not raise

    def test_a_missing_title_is_review_because_it_costs_attribution(self):
        path = write([("1", "", BODY)], self.directory)
        findings = corpus.check(corpus.read(path))
        titles = [f for f in findings if "no title" in f.problem]
        self.assertEqual(len(titles), 1)
        self.assertEqual(titles[0].severity, "review")

    def test_a_clean_corpus_produces_no_findings(self):
        path = write([("1", "One", BODY), ("2", "Two", BODY)], self.directory)
        self.assertEqual(corpus.check(corpus.read(path)), [])


class TestSection(unittest.TestCase):
    def test_metadata_is_what_travels_into_the_store(self):
        section = corpus.Section("4.2", "Identification of critical equipment", BODY)
        self.assertEqual(section.metadata(),
                         {"chapter": "4.2", "title": "Identification of critical equipment"})

    def test_the_citation_names_chapter_and_title(self):
        section = corpus.Section("5.4", "Emergency planning", BODY)
        self.assertEqual(section.citation(), "Chapter 5.4, Emergency planning")


class TestTheShippedCorpus(unittest.TestCase):
    """The real JRC guidance, as segmented for this system.

    Pinned, because the README quotes these figures and because a corpus that
    silently changes size invalidates every retrieval number measured against it.
    """

    @classmethod
    def setUpClass(cls):
        cls.sections = corpus.load(REAL)

    def test_it_loads_without_a_blocking_finding(self):
        self.assertEqual(
            [f for f in corpus.check(self.sections) if f.severity == "blocking"], [])

    def test_it_holds_forty_four_chapters(self):
        self.assertEqual(len(self.sections), 44)

    def test_the_size_matches_what_the_readme_states(self):
        stats = corpus.statistics(self.sections)
        self.assertEqual(stats["sections"], 44)
        self.assertEqual(stats["characters"], 107312)

    def test_every_section_has_a_chapter_and_a_title(self):
        for section in self.sections:
            self.assertTrue(section.chapter, section.title)
            self.assertTrue(section.title, section.chapter)

    def test_chapter_numbers_are_unique(self):
        chapters = [s.chapter for s in self.sections]
        self.assertEqual(len(chapters), len(set(chapters)))

    def test_the_nesting_of_the_guidance_survived_segmentation(self):
        """Chapters run from 1 to 5.5 with subsections like 4.6.2.2. Flattening
        that would lose the structure the metadata exists to preserve."""
        chapters = {s.chapter for s in self.sections}
        for expected in ("1", "3.3.1", "4.6.2.2", "5.5"):
            self.assertIn(expected, chapters)


if __name__ == "__main__":
    unittest.main()
