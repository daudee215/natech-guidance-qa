"""Tests for the groundedness screen.

The tests that matter here are the negative ones. A checker that says
SUPPORTED too readily is worse than no checker, because it converts an unread
answer into a checked one in the reader's mind. So most of what follows is
about the cases where it must refuse to give a pass: a number that is not in
the source, a number in the wrong chapter, and a claim with nothing checkable
in it, which must come back UNVERIFIABLE and never SUPPORTED.
"""

import unittest

from natech import groundedness as g
from natech.groundedness import (DERIVED, MISATTRIBUTED, SUPPORTED,
                                 UNSUPPORTED, UNVERIFIABLE, Evidence)


def ev(chapter, content, title="Section"):
    return Evidence(chapter=chapter, title=title, content=content)


class TestUnits(unittest.TestCase):
    def test_the_same_quantity_written_three_ways_is_one_token(self):
        forms = ["500 m", "500m", "500 metres"]
        values = {g.tokens(f)[0].value for f in forms}
        self.assertEqual(values, {"500m"})

    def test_a_thousands_separator_does_not_make_a_different_number(self):
        self.assertEqual(g.tokens("1,000 m")[0].value, g.tokens("1000 m")[0].value)

    def test_a_trailing_zero_does_not_make_a_different_number(self):
        self.assertEqual(g.tokens("500.0 m")[0].value, g.tokens("500 m")[0].value)

    def test_a_leading_decimal_point_is_read(self):
        self.assertEqual(g.tokens(".5 m")[0].value, "0.5m")

    def test_percent_spellings_agree(self):
        values = {g.tokens(f)[0].value for f in ["10%", "10 %", "10 percent"]}
        self.assertEqual(values, {"10%"})

    def test_the_unit_is_part_of_the_token(self):
        """500 m and 500 kg are not the same fact and must not match."""
        self.assertNotEqual(g.tokens("500 m")[0].value, g.tokens("500 kg")[0].value)

    def test_a_bare_number_does_not_match_a_number_with_a_unit(self):
        self.assertNotEqual(g.tokens("500")[0].value, g.tokens("500 m")[0].value)


class TestStructuralReferences(unittest.TestCase):
    def test_a_chapter_reference_is_not_a_claim(self):
        """Otherwise every correctly cited answer is flagged for its citation."""
        self.assertEqual(g.tokens("See Chapter 7."), [])

    def test_all_the_structural_words_are_excluded(self):
        for word in ("Section", "Table", "Figure", "Annex", "Appendix", "Step",
                     "Box", "page", "paragraph"):
            self.assertEqual(g.tokens("in {0} 4".format(word)), [],
                             "{0} 4 was treated as a factual claim".format(word))

    def test_a_real_quantity_next_to_a_reference_still_counts(self):
        found = g.tokens("Table 3 gives a separation of 500 m.")
        self.assertEqual([t.value for t in found], ["500m"])

    def test_an_abbreviated_reference_with_a_full_stop_is_excluded(self):
        self.assertEqual(g.tokens("see Fig. 12"), [])


class TestBareNumberNoise(unittest.TestCase):
    """Rules added after running the extractor over the real JRC corpus.

    Before these, 152 tokens came out of 44 chapters and almost all of the bare
    ones were citation years and list indices. After, 23 come out and they are
    quantities. The tests below pin each exclusion so a later loosening has to
    be deliberate.
    """

    def test_a_citation_year_is_not_a_quantity(self):
        self.assertEqual(g.tokens("as set out in INERIS, 2014, and OECD"), [])

    def test_a_small_bare_integer_is_treated_as_a_list_index(self):
        self.assertEqual(g.tokens("The procedure has 7 parts"), [])

    def test_a_year_with_a_unit_is_still_a_quantity(self):
        """'every 5 years' is a real interval and must survive."""
        self.assertEqual([t.value for t in g.tokens("repeated every 5 years")],
                         ["5y"])

    def test_a_large_bare_number_survives(self):
        """This is what catches an invented unitless threshold."""
        self.assertEqual([t.value for t in g.tokens("a score of 470")], ["470"])

    def test_a_small_number_with_a_unit_survives(self):
        self.assertEqual([t.value for t in g.tokens("within 5 m")], ["5m"])

    def test_a_plural_structural_word_is_excluded(self):
        """'Steps 1-6' was read as a named standard before plurals were
        stripped."""
        self.assertEqual(g.tokens("analyse (Steps 1-6) and evaluate"), [])


class TestStandardsAndDirectives(unittest.TestCase):
    def test_a_directive_is_one_token_not_three_numbers(self):
        found = g.tokens("under Directive 2012/18/EU")
        self.assertEqual([t.kind for t in found], ["directive"])

    def test_a_named_standard_is_captured(self):
        values = [t.value for t in g.tokens("ISO 31000 applies")]
        self.assertIn("ISO 31000", values)

    def test_seveso_with_a_roman_numeral(self):
        values = [t.value for t in g.tokens("the Seveso III Directive")]
        self.assertIn("SEVESO III", values)


class TestSplittingClaims(unittest.TestCase):
    def test_a_decimal_does_not_end_a_sentence(self):
        claims = g.split_claims("The limit is 1.5 m. It applies to all sites.")
        self.assertEqual(len(claims), 2)
        self.assertIn("1.5 m", claims[0])

    def test_an_abbreviation_does_not_end_a_sentence(self):
        claims = g.split_claims("Some plants, e.g. refineries, are in scope.")
        self.assertEqual(len(claims), 1)

    def test_empty_text_gives_no_claims(self):
        for text in ("", "   ", "\n\n"):
            self.assertEqual(g.split_claims(text), [])

    def test_a_numbered_list_marker_does_not_split(self):
        claims = g.split_claims("Do the following. 1. Assess the hazard.")
        self.assertTrue(any("Assess the hazard" in c for c in claims))
        self.assertNotIn("1.", [c.strip() for c in claims])


class TestVerdicts(unittest.TestCase):
    EVIDENCE = [
        ev("3", "Facilities shall maintain a minimum separation of 500 m from "
                "the site boundary, and the assessment shall be repeated every "
                "5 years under Directive 2012/18/EU."),
        ev("7", "Overpressure above 10 kPa is treated as causing structural "
                "damage to ordinary buildings."),
    ]

    def test_a_number_present_in_the_evidence_is_supported(self):
        report = g.check("A separation of 500 m is required.", self.EVIDENCE)
        self.assertEqual(report.verdict, SUPPORTED)

    def test_a_number_absent_from_the_evidence_is_unsupported(self):
        report = g.check("A separation of 800 m is required.", self.EVIDENCE)
        self.assertEqual(report.verdict, UNSUPPORTED)
        self.assertIn("800 m", str(report))

    def test_the_right_number_in_the_wrong_chapter_is_misattributed(self):
        """The claim is true. The citation still does not lead a reader to it."""
        report = g.check("Overpressure above 10 kPa causes damage.",
                         self.EVIDENCE, cited=["3"])
        self.assertEqual(report.verdict, MISATTRIBUTED)
        self.assertIn("Chapter 7", str(report))

    def test_a_claim_with_nothing_checkable_is_unverifiable_not_supported(self):
        """The single most important test in this file."""
        report = g.check("Operators should take a precautionary approach.",
                         self.EVIDENCE)
        self.assertEqual(report.verdict, UNVERIFIABLE)
        self.assertEqual(report.counts[SUPPORTED], 0)

    def test_an_invented_directive_is_caught(self):
        report = g.check("This falls under Directive 1999/92/EC.", self.EVIDENCE)
        self.assertEqual(report.verdict, UNSUPPORTED)

    def test_the_worst_verdict_wins(self):
        report = g.check(
            "A separation of 500 m is required. A separation of 800 m is also "
            "required.", self.EVIDENCE)
        self.assertEqual(report.counts[SUPPORTED], 1)
        self.assertEqual(report.counts[UNSUPPORTED], 1)
        self.assertEqual(report.verdict, UNSUPPORTED)

    def test_an_empty_answer_is_unverifiable(self):
        self.assertEqual(g.check("", self.EVIDENCE).verdict, UNVERIFIABLE)

    def test_no_evidence_at_all_makes_every_checkable_claim_unsupported(self):
        report = g.check("A separation of 500 m is required.", [])
        self.assertEqual(report.verdict, UNSUPPORTED)


class TestDerivation(unittest.TestCase):
    EVIDENCE = [ev("2", "Zone A extends 300 m and zone B extends 200 m from "
                        "the source.")]

    def test_a_sum_of_source_numbers_is_derived_not_fabricated(self):
        report = g.check("The total extent is 500 m.", self.EVIDENCE)
        self.assertEqual(report.verdict, DERIVED)
        self.assertIn("300 + 200", str(report))

    def test_a_difference_is_derived(self):
        report = g.check("The difference is 100 m.", self.EVIDENCE)
        self.assertEqual(report.verdict, DERIVED)

    def test_a_number_that_is_neither_present_nor_derivable_is_unsupported(self):
        report = g.check("The extent is 777 m.", self.EVIDENCE)
        self.assertEqual(report.verdict, UNSUPPORTED)

    def test_derivation_does_not_cross_units(self):
        """300 m + 200 m must not explain 500 kg."""
        report = g.check("The mass is 500 kg.", self.EVIDENCE)
        self.assertEqual(report.verdict, UNSUPPORTED)

    def test_derived_ranks_below_unsupported_in_the_overall_verdict(self):
        report = g.check("The total is 500 m. The limit is 777 m.", self.EVIDENCE)
        self.assertEqual(report.verdict, UNSUPPORTED)


class TestReport(unittest.TestCase):
    EVIDENCE = [ev("3", "The threshold is 500 m.")]

    def test_checkable_share_is_zero_when_nothing_could_be_checked(self):
        report = g.check("Operators should be careful. Guidance should be "
                         "followed.", self.EVIDENCE)
        self.assertEqual(report.checkable_share, 0.0)

    def test_checkable_share_distinguishes_a_quiet_run_from_a_clean_one(self):
        quiet = g.check("Operators should be careful.", self.EVIDENCE)
        clean = g.check("The threshold is 500 m.", self.EVIDENCE)
        self.assertEqual(quiet.verdict, UNVERIFIABLE)
        self.assertEqual(clean.verdict, SUPPORTED)
        self.assertLess(quiet.checkable_share, clean.checkable_share)

    def test_the_report_says_the_screen_is_not_entailment(self):
        """A clean report that does not say what it did not check invites the
        reader to over-trust it."""
        text = str(g.check("The threshold is 500 m.", self.EVIDENCE))
        self.assertIn("not entailment", text)

    def test_problems_lists_only_the_claims_needing_attention(self):
        report = g.check("The threshold is 500 m. The limit is 900 m. "
                         "Operators should be careful.", self.EVIDENCE)
        self.assertEqual(len(report.problems()), 1)
        self.assertEqual(report.problems()[0].verdict, UNSUPPORTED)

    def test_counts_sum_to_the_number_of_claims(self):
        report = g.check("The threshold is 500 m. The limit is 900 m. "
                         "Operators should be careful.", self.EVIDENCE)
        self.assertEqual(sum(report.counts.values()), len(report.claims))


if __name__ == "__main__":
    unittest.main()
