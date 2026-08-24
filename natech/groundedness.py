"""Checking that an answer is actually supported by the sections it cites.

A retrieval system fails in two very different ways. Retrieval can miss, which
retrieval.py measures. Or retrieval can succeed and the generated answer can
still say something the retrieved text does not support, cite the wrong chapter
for a claim that is otherwise true, or quietly do arithmetic nobody asked for.
That second class is invisible to a recall@k number and it is the class that
matters here, because the corpus is technical guidance and the numbers in it are
the part a reader would act on.

What this module does, and what it deliberately does not do
-----------------------------------------------------------
Deciding whether a sentence is entailed by a passage is natural language
inference and needs a model. There is no NLI model in this repository and a
lexical method dressed up as one would be worse than nothing, because it would
produce confident verdicts on exactly the paraphrases it cannot read.

So this is a screen, not a judge. It checks the class it can check with almost
no false negatives: numeric quantities with their units, and references to
named standards and directives. If an answer says "a minimum separation of
500 m" and no retrieved section contains 500 m, the model did not read it in
the sources. Paraphrase does not hide a number.

Everything else is reported as UNVERIFIABLE rather than passed. A claim with
nothing checkable in it gets no verdict, and the report says how much of the
answer fell into that bucket, so nobody mistakes a quiet run for a clean one.

The five verdicts
-----------------
NOT_CONTRADICTED  every checkable token in the claim appears in the cited
                  evidence. The name is the whole of what the test does: set
                  membership of numeric tokens. A negated claim, a substituted
                  subject and an invention with no numbers in it all pass it,
                  so calling the verdict SUPPORTED would be claiming an
                  entailment that nothing here established
MISATTRIBUTED     the tokens are in the corpus, but not in the section cited
                  for them. The claim may well be true; the citation still
                  does not lead a reader to it, which defeats the point of
                  citing
DERIVED           the number is not in any source but is arithmetic over
                  numbers that are. Not fabrication, but the model computed
                  something and nothing checked it
UNSUPPORTED       at least one checkable token appears nowhere in the evidence
UNVERIFIABLE      nothing checkable in this claim; no opinion offered

Standard library only, so this runs without Ollama, Chroma or a network.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

NOT_CONTRADICTED = "NOT_CONTRADICTED"
MISATTRIBUTED = "MISATTRIBUTED"
DERIVED = "DERIVED"
UNSUPPORTED = "UNSUPPORTED"
UNVERIFIABLE = "UNVERIFIABLE"

# Words that introduce a reference to the document's own structure. "Chapter 7"
# is the answer pointing at where it got something, not a factual claim of the
# quantity seven, and treating it as one would flag every correct citation.
STRUCTURAL = {
    "chapter", "section", "table", "figure", "fig", "annex", "appendix",
    "part", "page", "step", "box", "paragraph", "para", "clause", "item",
    "no", "number", "point", "line", "row", "column", "note", "footnote",
    # Observed in this corpus as a cross-reference: "see Approach 1 in
    # Figure 11". Kept narrow on purpose. "Tier 2" and "Category 3" are real
    # classifications in risk guidance and are deliberately not listed here.
    "approach",
}

# Unit spellings collapsed to one form, so "500 m", "500 metres" and "500m" are
# the same token. Order matters below: the longest spellings are tried first, or
# "m" would match the start of "metres".
UNIT_FORMS = [
    ("kilometres", "km"), ("kilometers", "km"), ("kilometre", "km"),
    ("kilometer", "km"), ("km2", "km2"), ("km²", "km2"), ("km/h", "km/h"),
    ("km", "km"),
    ("millimetres", "mm"), ("millimeters", "mm"), ("millimetre", "mm"),
    ("millimeter", "mm"), ("mm", "mm"),
    ("centimetres", "cm"), ("centimeters", "cm"), ("centimetre", "cm"),
    ("centimeter", "cm"), ("cm", "cm"),
    ("metres", "m"), ("meters", "m"), ("metre", "m"), ("meter", "m"),
    ("m2", "m2"), ("m²", "m2"), ("m3", "m3"), ("m³", "m3"), ("m", "m"),
    ("hectares", "ha"), ("hectare", "ha"), ("ha", "ha"),
    ("tonnes", "t"), ("tonne", "t"), ("kilotonnes", "kt"), ("kt", "kt"),
    ("megatonnes", "mt"), ("mt", "mt"), ("kilograms", "kg"),
    ("kilogram", "kg"), ("kg", "kg"),
    ("percent", "%"), ("per cent", "%"), ("%", "%"),
    ("degrees celsius", "°C"), ("°c", "°C"), ("°C", "°C"),
    ("kilopascals", "kPa"), ("kpa", "kPa"), ("megapascals", "MPa"),
    ("mpa", "MPa"), ("hectopascals", "hPa"), ("hpa", "hPa"),
    ("pascals", "Pa"), ("pa", "Pa"), ("bar", "bar"),
    ("seconds", "s"), ("second", "s"), ("minutes", "min"),
    ("minute", "min"), ("hours", "h"), ("hour", "h"),
    ("days", "d"), ("day", "d"), ("weeks", "wk"), ("week", "wk"),
    ("months", "mo"), ("month", "mo"), ("years", "y"), ("year", "y"),
    ("litres", "L"), ("liters", "L"), ("litre", "L"), ("liter", "L"),
    ("millilitres", "mL"), ("ml", "mL"),
    ("ppm", "ppm"), ("eur", "EUR"), ("euro", "EUR"), ("euros", "EUR"),
    ("€", "EUR"),
]

_UNIT_ALTERNATION = "|".join(
    re.escape(spelling) for spelling, _ in
    sorted(UNIT_FORMS, key=lambda pair: -len(pair[0])))

# A number, optionally with a unit immediately after it. Thousands separators
# and a leading decimal point are both allowed because both appear in the
# guidance text.
NUMBER = re.compile(
    r"(?<![\w.])"
    r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"\s*"
    r"(?P<unit>" + _UNIT_ALTERNATION + r")?"
    r"(?![\w])",
    re.IGNORECASE)

# EU legal instruments, "2012/18/EU" and the like. These are invented by models
# often enough to be worth their own pattern.
DIRECTIVE = re.compile(r"\b\d{4}\s*/\s*\d+\s*/\s*E[UC]\b", re.IGNORECASE)

# Named standards: a capitalised word or acronym followed by a number, such as
# "Seveso III", "ISO 31000", "ISO-31000", "EN 1998". The separator allows a
# hyphen because the guidance writes the same standard both ways, and matching
# only the spaced form would make one spelling checkable and the other not.
STANDARD = re.compile(
    r"\b([A-Z][A-Za-z]{1,14})[\s\-]((?:\d[\dA-Za-z.\-]*)|[IVXL]{1,6})\b")

# Bare numbers, meaning numbers with no unit, are excluded in two cases. Both
# were chosen after counting what the JRC corpus actually contains: of its 140
# extracted numbers, almost every bare one is either a citation year from a
# reference list or a list index, and neither is a factual quantity anybody
# would act on. Checking them adds noise and no protection.
#
# A bare number outside these ranges is kept. That is deliberate: it is what
# catches a fabricated unitless threshold, and it is also what surfaced the
# corrupted "ISO-310002" in chapter 3.
YEAR_RANGE = (1900, 2100)
LIST_INDEX_BELOW = 10

# Sentence boundary. Splitting on ". " alone breaks "1.5 m", "e.g." and
# "Chapter 7." so the lookbehind excludes a digit and the known abbreviations.
ABBREVIATIONS = {"e.g", "i.e", "cf", "etc", "vs", "fig", "no", "approx",
                 "ca", "al", "ref", "eq", "sec", "ch", "pp", "vol"}
_SENTENCE_END = re.compile(r"(?<=[.!?])[\s\n]+")

# Above this many source numbers sharing a unit the pairwise derivation search
# is skipped. 120 numbers is 7,140 pairs, which is fine; the cap exists so the
# cost stays bounded on a corpus larger than this one rather than because this
# one needs it.
DERIVATION_LIMIT = 120


def canonical_unit(spelling: Optional[str]) -> str:
    """Collapse a unit spelling to one form. Empty string for a bare number."""
    if not spelling:
        return ""
    lowered = spelling.strip().lower()
    for form, canonical in UNIT_FORMS:
        if lowered == form.lower():
            return canonical
    return lowered


@dataclass(frozen=True)
class Token:
    """One checkable thing: a quantity, a directive, or a standard reference."""

    kind: str          # "number" | "directive" | "standard"
    value: str         # normalised form, used for comparison
    surface: str       # what actually appeared in the text, for the report

    def __str__(self):
        return self.surface


def _is_structural(word: str) -> bool:
    """Is this the word that turns a following number into a pointer?

    Plurals are stripped because the guidance writes both "Step 7" and
    "Steps 1-6", and only catching the singular let "Steps 1-6" through as a
    named standard.
    """
    lowered = word.lower().rstrip(".")
    return lowered in STRUCTURAL or lowered.rstrip("s") in STRUCTURAL


def _is_noise(value: float, raw: str) -> bool:
    """A bare number that is almost certainly a year or a list index."""
    if value.is_integer():
        if YEAR_RANGE[0] <= value <= YEAR_RANGE[1] and len(raw) == 4:
            return True
        if 0 <= value < LIST_INDEX_BELOW:
            return True
    return False


def _number_spans(text: str):
    for match in NUMBER.finditer(text):
        # Skip the number if the word before it makes it a structural pointer.
        prefix = text[max(0, match.start() - 24):match.start()]
        words = re.findall(r"[A-Za-z]+", prefix)
        if words and _is_structural(words[-1]):
            continue
        raw = match.group("value").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        unit = canonical_unit(match.group("unit"))
        if not unit and _is_noise(value, raw):
            continue
        # Normalise 500.0 and 500 to the same key, so the same quantity written
        # two ways in answer and source still matches.
        normalised = "{0:g}{1}".format(value, unit)
        yield match.start(), match.end(), Token(
            "number", normalised, match.group(0).strip())


def _directive_spans(text: str):
    for match in DIRECTIVE.finditer(text):
        yield match.start(), match.end(), Token(
            "directive", re.sub(r"\s+", "", match.group(0)).upper(), match.group(0))


def _standard_spans(text: str):
    for match in STANDARD.finditer(text):
        name, number = match.group(1), match.group(2)
        if _is_structural(name):
            continue
        yield match.start(), match.end(), Token(
            "standard", "{0} {1}".format(name.upper(), number.upper()),
            match.group(0))


def tokens(text: str) -> List[Token]:
    """Every checkable token in a piece of text, in order of appearance.

    Overlaps are resolved by span, not by comparing surface strings. The three
    patterns genuinely do overlap: "Directive 2012/18/EU" matches the directive
    pattern once, the standard pattern as "Directive 2012", and the number
    pattern on each of its parts. Counting the same reference three times would
    let a source that writes it slightly differently fail one of the copies and
    report a false UNSUPPORTED.
    """
    claimed: List[Tuple[int, int]] = []
    found: List[Tuple[int, Token]] = []
    # Most specific pattern first: whatever it claims, the looser patterns are
    # not allowed to claim again.
    for spans in (_directive_spans(text), _standard_spans(text),
                  _number_spans(text)):
        for start, end, token in spans:
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            found.append((start, token))
    return [token for _, token in sorted(found, key=lambda pair: pair[0])]


def split_claims(text: str) -> List[str]:
    """Split an answer into sentence-level claims.

    Not a general sentence splitter. It needs to survive decimals, the
    abbreviations that appear in this corpus, and numbered list markers, which
    is the whole of what it is tested against.
    """
    if not text or not text.strip():
        return []
    pieces = []
    for piece in _SENTENCE_END.split(text.strip()):
        piece = piece.strip()
        if not piece:
            continue
        # Rejoin a piece that was split after an abbreviation or a bare number,
        # both of which end in "." without ending a sentence.
        if pieces:
            tail = re.findall(r"[A-Za-z0-9.]+", pieces[-1])
            last = tail[-1].rstrip(".").lower() if tail else ""
            if last in ABBREVIATIONS or re.fullmatch(r"\d+", last):
                pieces[-1] = pieces[-1] + " " + piece
                continue
        pieces.append(piece)
    return pieces


@dataclass(frozen=True)
class Evidence:
    """One retrieved section, as the checker needs it."""

    chapter: str
    title: str
    content: str

    def citation(self) -> str:
        return "Chapter {0}, {1}".format(self.chapter, self.title)


def _derivation(target: float, unit: str, pool: Sequence[float]) -> Optional[str]:
    """Explain target as arithmetic over pool, or None.

    Deliberately shallow: sums, differences, products of two source numbers,
    and one number as a percentage of another. Deeper search would start
    finding coincidences, and a coincidence reported as a derivation is worse
    than no explanation at all.
    """
    if len(pool) > DERIVATION_LIMIT:
        return None
    tolerance = max(abs(target) * 1e-6, 1e-9)
    for i, a in enumerate(pool):
        for b in pool[i:]:
            if abs(a + b - target) <= tolerance:
                return "{0:g} + {1:g}".format(a, b)
            if abs(abs(a - b) - target) <= tolerance and a != b:
                return "{0:g} - {1:g}".format(max(a, b), min(a, b))
            if abs(a * b - target) <= tolerance and a not in (0.0, 1.0) and b not in (0.0, 1.0):
                return "{0:g} x {1:g}".format(a, b)
    if unit == "%":
        for a in pool:
            for b in pool:
                if b and abs((a / b) * 100.0 - target) <= max(target * 1e-4, 1e-9):
                    return "{0:g} / {1:g} as a percentage".format(a, b)
    return None


@dataclass
class ClaimResult:
    """One claim, its verdict, and why."""

    claim: str
    verdict: str
    checked: List[Token] = field(default_factory=list)
    missing: List[Token] = field(default_factory=list)
    elsewhere: Dict[str, List[str]] = field(default_factory=dict)
    derivations: Dict[str, str] = field(default_factory=dict)

    def __str__(self):
        head = "[{0}] {1}".format(self.verdict, self.claim)
        notes = []
        for token in self.missing:
            where = self.elsewhere.get(token.value)
            how = self.derivations.get(token.value)
            if where:
                notes.append("      {0!r} is not in the cited section; it is in {1}"
                             .format(token.surface, "; ".join(where)))
            elif how:
                notes.append("      {0!r} is in no section, but equals {1}"
                             .format(token.surface, how))
            else:
                notes.append("      {0!r} appears in no retrieved section"
                             .format(token.surface))
        return "\n".join([head] + notes)


@dataclass
class Report:
    """The audit of one answer."""

    question: str
    claims: List[ClaimResult] = field(default_factory=list)
    evidence_sections: int = 0

    @property
    def counts(self) -> Dict[str, int]:
        result = {v: 0 for v in (NOT_CONTRADICTED, MISATTRIBUTED, DERIVED,
                                UNSUPPORTED, UNVERIFIABLE)}
        for claim in self.claims:
            result[claim.verdict] += 1
        return result

    @property
    def verdict(self) -> str:
        """The worst thing found, because that is what a reader needs to know."""
        counts = self.counts
        if counts[UNSUPPORTED]:
            return UNSUPPORTED
        if counts[MISATTRIBUTED]:
            return MISATTRIBUTED
        if counts[DERIVED]:
            return DERIVED
        if counts[NOT_CONTRADICTED]:
            return NOT_CONTRADICTED
        return UNVERIFIABLE

    @property
    def checkable_share(self) -> float:
        """Share of claims this method had any opinion on at all.

        Reported prominently. A run where every claim came back UNVERIFIABLE is
        not a clean run, it is a run where the screen saw nothing, and the two
        must never look the same in a log.
        """
        if not self.claims:
            return 0.0
        opinionated = len(self.claims) - self.counts[UNVERIFIABLE]
        return round(opinionated / float(len(self.claims)), 3)

    def problems(self) -> List[ClaimResult]:
        return [c for c in self.claims
                if c.verdict in (UNSUPPORTED, MISATTRIBUTED, DERIVED)]

    def __str__(self):
        counts = self.counts
        lines = [
            "Groundedness: {0}".format(self.verdict),
            "  question          {0}".format(self.question),
            "  sections cited    {0}".format(self.evidence_sections),
            "  claims            {0}".format(len(self.claims)),
            "  not contradicted  {0}".format(counts[NOT_CONTRADICTED]),
            "  misattributed     {0}".format(counts[MISATTRIBUTED]),
            "  derived           {0}".format(counts[DERIVED]),
            "  unsupported       {0}".format(counts[UNSUPPORTED]),
            "  unverifiable      {0}".format(counts[UNVERIFIABLE]),
            "  checkable share   {0}".format(self.checkable_share),
        ]
        problems = self.problems()
        if problems:
            lines += ["", "  needs a look:"]
            lines += ["    " + str(c).replace("\n", "\n  ") for c in problems]
        else:
            lines += ["", "  Nothing checkable was contradicted by the evidence.",
                      "  This is a lexical screen, not entailment: a claim can be",
                      "  wrong in ways it cannot see."]
        return "\n".join(lines)


def check(answer_text: str,
          evidence: Sequence[Evidence],
          cited: Optional[Sequence[str]] = None) -> Report:
    """Audit an answer against the sections it was generated from.

    `evidence` is everything retrieved. `cited` optionally narrows which
    chapters the answer actually pointed at; when given, a token found in the
    evidence but outside those chapters is MISATTRIBUTED rather than supported,
    because a citation a reader cannot follow to the claim is not a citation.
    """
    report = Report(question="", evidence_sections=len(evidence))

    # Token index per chapter, plus a flat set for the "is it anywhere" test.
    per_chapter: Dict[str, Set[str]] = {}
    numbers_by_unit: Dict[str, List[float]] = {}
    for section in evidence:
        found = tokens(section.content)
        per_chapter.setdefault(section.chapter, set()).update(t.value for t in found)
        for token in found:
            if token.kind != "number":
                continue
            match = re.fullmatch(r"(-?[\d.]+)(.*)", token.value)
            if match:
                try:
                    numbers_by_unit.setdefault(match.group(2), []).append(
                        float(match.group(1)))
                except ValueError:
                    pass

    cited_set = set(cited) if cited else set(per_chapter)
    in_cited: Set[str] = set()
    for chapter in cited_set:
        in_cited |= per_chapter.get(chapter, set())
    anywhere: Set[str] = set()
    for values in per_chapter.values():
        anywhere |= values

    for claim in split_claims(answer_text):
        checkable = tokens(claim)
        if not checkable:
            report.claims.append(ClaimResult(claim, UNVERIFIABLE))
            continue

        missing, elsewhere, derivations = [], {}, {}
        for token in checkable:
            if token.value in in_cited:
                continue
            missing.append(token)
            if token.value in anywhere:
                elsewhere[token.value] = sorted(
                    "Chapter {0}".format(chapter)
                    for chapter, values in per_chapter.items()
                    if token.value in values)
            elif token.kind == "number":
                match = re.fullmatch(r"(-?[\d.]+)(.*)", token.value)
                if match:
                    how = _derivation(float(match.group(1)), match.group(2),
                                      numbers_by_unit.get(match.group(2), []))
                    if how:
                        derivations[token.value] = how

        if not missing:
            verdict = NOT_CONTRADICTED
        elif all(t.value in elsewhere for t in missing):
            verdict = MISATTRIBUTED
        elif all(t.value in elsewhere or t.value in derivations for t in missing):
            verdict = DERIVED
        else:
            verdict = UNSUPPORTED

        report.claims.append(ClaimResult(
            claim, verdict, checked=checkable, missing=missing,
            elsewhere=elsewhere, derivations=derivations))

    return report
