"""The vector store: embedding the guidance locally and retrieving from it.

This is the April code, kept as it was in shape and in its choices. Everything
here needs Ollama running and the langchain packages installed, which is why it
is a separate module from corpus.py: the loading and validation that decide what
gets indexed stay checkable without a model server.

Two things were changed in August and both are noted where they happen. The
imports are deferred to call time so the package can be imported, and its corpus
layer tested, on a machine with none of this installed.

Why local rather than an API, which was the original decision and still holds:
the corpus stays off third-party infrastructure, and the retrieval step is
inspectable, which matters more than throughput at this stage.
"""

import os
from typing import List, Optional, Sequence

from .corpus import Section, load

DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_COLLECTION = "Natech_risk_management"
DEFAULT_STORE_PATH = "./chroma_store"
DEFAULT_K = 2


class BackendMissing(ImportError):
    """langchain, Chroma or Ollama is not available."""


def _imports():
    """Import the backend, or explain what to install.

    Deferred so that `import natech` works with nothing installed. The message
    names the packages rather than letting an ImportError surface three frames
    down inside langchain.
    """
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_ollama import OllamaEmbeddings
        return Chroma, Document, OllamaEmbeddings
    except ImportError as error:
        raise BackendMissing(
            "the vector store needs langchain-chroma and langchain-ollama, and "
            "an Ollama server with the {0} model pulled. Install with "
            "pip install '.[rag]' and run: ollama pull {0}. "
            "The corpus layer in natech.corpus needs none of this. "
            "Original error: {1}".format(DEFAULT_EMBEDDING_MODEL, error))


def documents_from(sections: Sequence[Section]):
    """One Document per section, carrying chapter and title as metadata.

    Carrying them as metadata rather than concatenating them into the text costs
    nothing at build time and is the difference between an answer that can be
    attributed to a named section and an answer that cannot.
    """
    _, Document, _ = _imports()
    return [
        Document(page_content=section.content,
                 metadata=section.metadata(),
                 id=section.identifier)
        for section in sections
    ]


def build(corpus_path: str = "data/guidance.csv",
          store_path: str = DEFAULT_STORE_PATH,
          collection: str = DEFAULT_COLLECTION,
          embedding_model: str = DEFAULT_EMBEDDING_MODEL,
          rebuild: bool = False):
    """Build or reuse the persistent store, and return it.

    Reuse is gated on the store already holding the expected number of
    documents, not merely on the directory existing.

    The April version tested `os.path.exists(store_path)`. That is right almost
    always and wrong in the case that costs the most: a build interrupted part
    way leaves the directory in place holding some of the corpus, and every run
    afterwards reuses it. The store then answers from a subset of the guidance
    and nothing anywhere reports a problem. Counting is cheap and catches it.
    """
    Chroma, _, OllamaEmbeddings = _imports()
    sections = load(corpus_path)
    embeddings = OllamaEmbeddings(model=embedding_model)
    store = Chroma(collection_name=collection,
                   persist_directory=store_path,
                   embedding_function=embeddings)

    existing = 0
    if os.path.exists(store_path):
        try:
            existing = len(store.get().get("ids", []))
        except Exception:
            existing = 0

    if rebuild or existing != len(sections):
        if existing:
            print("store holds {0} of {1} sections, rebuilding".format(
                existing, len(sections)))
        documents = documents_from(sections)
        store.add_documents(documents=documents,
                            ids=[s.identifier for s in sections])
    return store


def retriever(store=None, k: int = DEFAULT_K, **kwargs):
    """A retriever over the store.

    Exposed as a single object so the generation side imports a retriever rather
    than a vector store, and the store can be replaced by a graph-backed or
    workflow-aware retriever without the generation code changing.

    k defaults to 2, which is the April value and a low ceiling. Two passages
    cannot answer a question that needs to combine two parts of the guidance.
    It is a parameter rather than a constant so retrieval.py can measure what
    raising it buys.
    """
    if store is None:
        store = build(**kwargs)
    return store.as_retriever(search_kwargs={"k": k})
