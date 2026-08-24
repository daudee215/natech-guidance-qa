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
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

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
    sections: tuple          # (chapter, title) pairs, in retrieval order
    model: str

    def citations(self) -> List[str]:
        return ["Chapter {0}, {1}".format(c, t) for c, t in self.sections]

    def __str__(self):
        if not self.sections:
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
    sections = tuple(
        (d.metadata.get("chapter", ""), d.metadata.get("title", ""))
        for d in documents
    )
    text = _chain(model_name).invoke({"contents": documents, "question": question})
    return Answer(question=question, text=str(text).strip(),
                  sections=sections, model=model_name)
