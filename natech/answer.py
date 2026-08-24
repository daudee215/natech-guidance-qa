"""Generation: the retrieved sections, the prompt, and the model.

The April code, kept. The prompt is the one that was written for this corpus and
is not rewritten here, because it is part of what the system is and changing it
silently would make the retrieval numbers in retrieval.py incomparable with
anything measured before.

What was added in August is the citation. The April version passed the retrieved
Documents into the prompt and printed whatever came back, which meant an answer
could not be traced to a section without going back to the retriever by hand.
The chapter and title were already travelling with every passage; they just were
not being used. `answer()` now returns the sections it used alongside the text.

What was added after that is the evidence itself. Returning only the chapter and
title still made the answer unauditable: knowing that an answer came from
Chapter 7 does not tell anybody whether Chapter 7 says it. The retrieved text
was being read into the prompt and then dropped on the floor. It is now kept, so
`Answer.audit()` can put the generated text back against the passages it was
generated from. See groundedness.py for what that check does and does not claim.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .groundedness import Evidence, Report, check
from .store import BackendMissing, DEFAULT_K

DEFAULT_MODEL = "llama3.2"

# The April prompt, unchanged.
TEMPLATE = """
You are an expert in answering questions about Natech risk management and Resilience of High-TECH industries and Critical infrastructures across Europe

Here are some relevant contents: {contents}

Here is the question to answer: {question}
"""


@dataclass(frozen=True)
class Answer:
    """What the system produced, and what it produced it from."""

    question: str
    text: str
    evidence: tuple          # Evidence objects, in retrieval order
    model: str

    @property
    def sections(self) -> tuple:
        """(chapter, title) pairs. Derived, so the evidence stays the one
        source of truth about what was retrieved."""
        return tuple((e.chapter, e.title) for e in self.evidence)

    def citations(self) -> List[str]:
        return [e.citation() for e in self.evidence]

    def audit(self, cited: Optional[Sequence[str]] = None) -> Report:
        """Check the generated text against the passages it came from.

        A screen over quantities and named standards, not entailment. A clean
        result means nothing checkable was contradicted, which is weaker than
        the answer being right, and groundedness.py says so at length.
        """
        report = check(self.text, self.evidence, cited=cited)
        report.question = self.question
        return report

    def __str__(self):
        if not self.evidence:
            return self.text + "\n\n(no sections retrieved)"
        return "{0}\n\nFrom: {1}".format(self.text, "; ".join(self.citations()))


def _chain(model_name: str):
    """Build the prompt-to-model chain. Deferred import, as in store.py."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama.llms import OllamaLLM
    except ImportError as error:
        raise BackendMissing(
            "generation needs langchain-core and langchain-ollama, and an "
            "Ollama server with {0} pulled. Install with pip install '.[rag]' "
            "and run: ollama pull {0}. Original error: {1}".format(
                model_name, error))
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    return prompt | OllamaLLM(model=model_name)


def answer(question: str, retriever, model_name: str = DEFAULT_MODEL) -> Answer:
    """Retrieve, generate, and report which sections the answer came from.

    Retrieval failing and generation failing look the same from the outside: a
    poor answer. Returning the sections separates them, which is the whole
    reason chapter and title are carried through as metadata.
    """
    documents = retriever.invoke(question)
    evidence = tuple(
        Evidence(chapter=d.metadata.get("chapter", ""),
                 title=d.metadata.get("title", ""),
                 content=getattr(d, "page_content", ""))
        for d in documents
    )
    text = _chain(model_name).invoke({"contents": documents, "question": question})
    return Answer(question=question, text=str(text).strip(),
                  evidence=evidence, model=model_name)
