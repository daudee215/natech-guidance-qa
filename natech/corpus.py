"""Loading the segmented guidance, and refusing to index a corpus that is broken.

The corpus is the JRC technical guidance on Natech risk management, segmented by
chapter into 44 records with three fields: chapter number, section title, body
text. Segmentation is at chapter granularity and the structure is kept rather
than flattened, so chapter and title travel with the text through embedding and
retrieval. An answer is then attributable to a named section, and a failure is
diagnosable as retrieval or as generation rather than as one undifferentiated
mistake.

This module is standard library only. Everything that needs Ollama or Chroma
lives in store.py, so the loading and validation that decide what goes into the
index can be checked without a model server running.

The validation is the part added in August. The original loader took whatever
the CSV held: a row with an empty Content field became a Document with empty
page_content, which embeds to something meaningless, sits in the store, and can
be retrieved as context for a question it says nothing about. Nothing raised and
nothing logged. That is the failure this module exists to stop.
"""

import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

# The three columns the segmentation produces. Named here rather than assumed at
# each use site, so a renamed column fails once with a clear message.
CHAPTER, TITLE, CONTENT = "Chapter", "Title", "Content"

# Below this a record is almost certainly a heading that was captured without
# its body, which is the defect the chapter-level segmentation actually produces.
MINIMUM_CONTENT_CHARACTERS = 120


class CorpusError(ValueError):
    """The corpus cannot be indexed as it stands."""


@dataclass(frozen=True)
class Section:
    """One chapter-level record of the guidance."""

    chapter: str
    title: str
    content: str

    @property
    def identifier(self) -> str:
        return self.chapter

    @property
    def characters(self) -> int:
        return len(self.content)

    def citation(self) -> str:
        """How this section should be named in an answer."""
        return "Chapter {0}, {1}".format(self.chapter, self.title)

    def metadata(self) -> Dict[str, str]:
        """What travels with the text into the vector store."""
        return {"chapter": self.chapter, "title": self.title}


@dataclass(frozen=True)
class Finding:
    """One thing wrong with the corpus."""

    row: int
    chapter: str
    problem: str
    severity: str = "blocking"

    def __str__(self):
        return "[{0}] row {1} (chapter {2}): {3}".format(
            self.severity, self.row, self.chapter or "?", self.problem)


def read(path: str) -> List[Section]:
    """Read the segmented guidance from CSV.

    Does not validate. `load` is the function to call; this one is separate so
    that `check` can report on a corpus that `load` would refuse.
    """
    if not os.path.exists(path):
        raise CorpusError("no corpus at {0}".format(path))
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {CHAPTER, TITLE, CONTENT} - set(reader.fieldnames or [])
        if missing:
            raise CorpusError(
                "corpus is missing column(s) {0}; found {1}".format(
                    ", ".join(sorted(missing)), ", ".join(reader.fieldnames or [])))
        return [
            Section(chapter=(row.get(CHAPTER) or "").strip(),
                    title=(row.get(TITLE) or "").strip(),
                    content=(row.get(CONTENT) or "").strip())
            for row in reader
        ]


def check(sections: Sequence[Section],
          minimum_characters: int = MINIMUM_CONTENT_CHARACTERS) -> List[Finding]:
    """Everything wrong with this corpus, worst first."""
    findings: List[Finding] = []
    seen: Dict[str, int] = {}

    for index, section in enumerate(sections, start=1):
        if not section.chapter:
            findings.append(Finding(index, "", "no chapter number"))
        elif section.chapter in seen:
            findings.append(Finding(
                index, section.chapter,
                "duplicate chapter number, first seen at row {0}. Documents are "
                "keyed on it, so one would overwrite the other in the store"
                .format(seen[section.chapter])))
        else:
            seen[section.chapter] = index

        if not section.title:
            findings.append(Finding(
                index, section.chapter,
                "no title. The title is carried as metadata and is what an "
                "answer cites, so a section without one cannot be attributed",
                severity="review"))

        if not section.content:
            findings.append(Finding(
                index, section.chapter,
                "empty content. This would embed to nothing and still be "
                "retrievable as context"))
        elif section.characters < minimum_characters:
            findings.append(Finding(
                index, section.chapter,
                "only {0} characters, below the {1} character floor. Usually a "
                "heading captured without its body".format(
                    section.characters, minimum_characters),
                severity="review"))

    return findings


def load(path: str, minimum_characters: int = MINIMUM_CONTENT_CHARACTERS) -> List[Section]:
    """Read the corpus and refuse to return it if anything blocking is wrong.

    Refusing is the point. A retrieval system built on a corpus with holes in it
    answers questions confidently from the sections that survived, and the
    sections that did not are invisible from the outside.
    """
    sections = read(path)
    if not sections:
        raise CorpusError("corpus at {0} has no rows".format(path))
    findings = check(sections, minimum_characters=minimum_characters)
    blocking = [f for f in findings if f.severity == "blocking"]
    if blocking:
        raise CorpusError(
            "corpus at {0} has {1} blocking problem(s):\n{2}".format(
                path, len(blocking), "\n".join("  " + str(f) for f in blocking)))
    return sections


def statistics(sections: Sequence[Section]) -> dict:
    """Size and shape of the corpus, for the README and for a sanity check."""
    if not sections:
        return {"sections": 0, "characters": 0, "shortest": 0, "longest": 0,
                "mean": 0.0}
    lengths = [s.characters for s in sections]
    return {
        "sections": len(sections),
        "characters": sum(lengths),
        "shortest": min(lengths),
        "longest": max(lengths),
        "mean": round(sum(lengths) / float(len(lengths)), 1),
    }


def report(sections: Sequence[Section], findings: Sequence[Finding]) -> str:
    """The corpus as text, for a log or a data statement."""
    stats = statistics(sections)
    lines = [
        "Corpus",
        "  sections          {0}".format(stats["sections"]),
        "  characters        {0}".format(stats["characters"]),
        "  shortest section  {0}".format(stats["shortest"]),
        "  longest section   {0}".format(stats["longest"]),
        "  mean              {0}".format(stats["mean"]),
    ]
    blocking = [f for f in findings if f.severity == "blocking"]
    lines.append("  safe to index     {0}".format("no" if blocking else "yes"))
    if findings:
        lines += ["", "  findings:"]
        lines += ["    " + str(f) for f in findings]
    return "\n".join(lines)
