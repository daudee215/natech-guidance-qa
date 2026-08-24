"""Measuring retrieval instead of reading it.

Added in August. The April system had no evaluation: quality was judged by
reading the passages that came back and deciding they looked relevant. That is
how retrieval quality is usually judged and it has two problems. It does not
survive a change to the chunking or the embedding model, because there is
nothing to compare against. And it cannot answer the question that actually
matters here, which is whether returning two passages is enough.

What this module measures is recall at k: given a question and the chapter of
the guidance that should answer it, does the retriever return that chapter in
its top k. That is a weak measure of a retrieval system and a sufficient one for
the decision in front of it. It does not measure whether the answer generated
from those passages was any good, and nothing here claims it does.

The evaluation takes a retrieval function rather than a store, so it runs
against the real retriever, against a keyword baseline, or against a stub in the
tests. That is also what makes the baseline comparison below possible: a
retrieval system that cannot beat term overlap is not earning its embeddings.
"""

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .corpus import Section

# A retrieval function takes a question and a k, and returns chapter identifiers
# in rank order.
Retrieve = Callable[[str, int], Sequence[str]]


@dataclass(frozen=True)
class Probe:
    """One question, and the chapters that should answer it.

    `expected` holds chapter identifiers. More than one is allowed, because
    several questions here are genuinely answered by more than one section and
    forcing a single gold chapter would score a correct retrieval as wrong.
    """

    question: str
    expected: tuple
    note: str = ""


# Written by hand against the chapter titles of this specific corpus. This is a
# probe set, not a benchmark: 12 questions chosen to cover the document rather
# than sampled from real user questions, and the expected chapters are my
# judgement of where the answer lives. Its purpose is to make a change to the
# chunking or to k measurable, not to report a quality score for the system.
# Real questions from real users would replace it and would probably be harder.
PROBES = (
    Probe("What is a Natech accident?", ("1",),
          "the definition, which should be the easiest retrieval in the set"),
    Probe("Why are Natech events harder to manage than ordinary industrial accidents?",
          ("2",)),
    Probe("How should an operator communicate about Natech risk with authorities?",
          ("3.1",)),
    Probe("Which items of equipment should be treated as critical?", ("4.2",)),
    Probe("How is damage from a natural hazard to process equipment characterised?",
          ("4.3",)),
    Probe("What role do safety barriers and utilities play as contributing factors?",
          ("4.4", "5.3"), "answered by two sections, so both count"),
    Probe("How is loss of containment modelled after equipment damage?",
          ("4.6.1", "4.6.1.1")),
    Probe("What happens when several accidents occur at the same time?",
          ("4.6.2.2",)),
    Probe("How is the likelihood of an indirect Natech accident estimated?",
          ("4.7.1.3",)),
    Probe("What physical measures reduce the impact of a natural hazard?",
          ("5.1.1",)),
    Probe("What should be done in emergency planning for a Natech accident?",
          ("5.4",)),
    Probe("What can be learned from past Natech accidents?", ("5.5",)),
)


@dataclass(frozen=True)
class ProbeResult:
    """How one probe went."""

    probe: Probe
    returned: tuple
    hit: bool
    rank: Optional[int]

    def __str__(self):
        where = "rank {0}".format(self.rank) if self.hit else "not returned"
        return "{0:<62} {1}".format(self.probe.question[:60], where)


def evaluate(retrieve: Retrieve, probes: Sequence[Probe] = PROBES,
             k: int = 2) -> dict:
    """Recall at k and mean reciprocal rank over the probe set."""
    results: List[ProbeResult] = []
    for probe in probes:
        returned = tuple(retrieve(probe.question, k))
        rank = None
        for position, chapter in enumerate(returned, start=1):
            if chapter in probe.expected:
                rank = position
                break
        results.append(ProbeResult(probe, returned, rank is not None, rank))

    hits = [r for r in results if r.hit]
    return {
        "k": k,
        "probes": len(results),
        "hits": len(hits),
        "recall_at_k": round(len(hits) / float(len(results)), 4) if results else None,
        "mrr": (round(sum(1.0 / r.rank for r in hits) / len(results), 4)
                if results else None),
        "results": tuple(results),
        "missed": tuple(r.probe.question for r in results if not r.hit),
    }


def sweep(retrieve: Retrieve, probes: Sequence[Probe] = PROBES,
          ks: Sequence[int] = (1, 2, 3, 5, 8)) -> List[dict]:
    """What raising k buys, measured rather than assumed.

    The question the April system could not answer. Retrieving more passages
    always weakly raises recall and always costs context, and the point at which
    recall stops improving is where k should sit.
    """
    return [evaluate(retrieve, probes, k=k) for k in ks]


# ------------------------------------------------------------------ baseline

_WORD = re.compile(r"[a-z]{3,}")
_COMMON = {
    "the", "and", "for", "are", "with", "that", "this", "from", "how", "what",
    "should", "which", "when", "does", "can", "was", "were", "has", "have",
    "been", "its", "their", "not", "but", "all", "any", "one", "two", "may",
    "natech", "risk", "accident", "accidents", "hazard", "hazards",
}


def keyword_retriever(sections: Sequence[Section]) -> Retrieve:
    """A term-overlap baseline, standard library only.

    Here so the embedding retriever has something to beat. A dense retriever
    that does not beat counting shared words is not paying for the model server
    it needs, and that comparison is not usually made because the baseline is
    not usually built.

    Deliberately crude: lowercase word overlap weighted by inverse document
    frequency, no stemming and no phrase matching. Natech-specific terms are
    stopped as well as English ones, since every section is about Natech risk
    and those words separate nothing.
    """
    corpus_terms = []
    frequency: Dict[str, int] = {}
    for section in sections:
        terms = {w for w in _WORD.findall(section.content.lower())
                 if w not in _COMMON}
        corpus_terms.append((section.identifier, terms))
        for term in terms:
            frequency[term] = frequency.get(term, 0) + 1

    total = float(len(sections)) or 1.0

    def retrieve(question: str, k: int) -> List[str]:
        asked = {w for w in _WORD.findall(question.lower()) if w not in _COMMON}
        scored = []
        for identifier, terms in corpus_terms:
            shared = asked & terms
            if not shared:
                continue
            score = sum(math.log(total / frequency[t]) for t in shared)
            scored.append((score, identifier))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [identifier for _, identifier in scored[:k]]

    return retrieve


def from_langchain(retriever) -> Retrieve:
    """Adapt a langchain retriever to the interface this module evaluates.

    The chapter comes back off the metadata, which is the reason the metadata is
    carried in the first place.
    """
    def retrieve(question: str, k: int) -> List[str]:
        try:
            documents = retriever.invoke(question, k=k)
        except TypeError:
            documents = retriever.invoke(question)
        return [d.metadata.get("chapter", "") for d in documents][:k]
    return retrieve


def report(results: Sequence[dict]) -> str:
    """A sweep as text."""
    if not results:
        return "Retrieval\n  nothing measured"
    lines = ["Retrieval, recall at k over {0} probes".format(results[0]["probes"]),
             "", "    k   recall    MRR   hits"]
    for result in results:
        lines.append("  {0:>3}   {1:>6.0%}  {2:>5.2f}   {3}/{4}".format(
            result["k"], result["recall_at_k"], result["mrr"],
            result["hits"], result["probes"]))
    best = max(r["recall_at_k"] for r in results)
    first_best = min(r["k"] for r in results if r["recall_at_k"] >= best)
    largest = max(r["k"] for r in results)
    lines.append("")
    if first_best == largest:
        # Still climbing at the largest k tried, so the plateau was not found.
        # Reporting a plateau here would be reporting the edge of the sweep.
        lines += [
            "  Recall is still improving at k = {0}, the largest tried, so this "
            "sweep".format(largest),
            "  has not found where it levels off. The honest reading is that k is "
            "still",
            "  the binding constraint, not that {0} is the right value.".format(largest),
        ]
    else:
        lines += [
            "  Recall stops improving at k = {0}. Retrieving more than that spends"
            .format(first_best),
            "  context without finding anything new on this probe set.",
        ]
    missed = results[-1]["missed"]
    if missed:
        lines += ["", "  Never retrieved, even at k = {0}:".format(results[-1]["k"])]
        lines += ["    " + q for q in missed]
    return "\n".join(lines)
