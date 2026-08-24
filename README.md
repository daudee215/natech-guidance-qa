# natech-guidance-qa

[![tests](https://github.com/daudee215/natech-guidance-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/daudee215/natech-guidance-qa/actions/workflows/tests.yml)

Retrieval-grounded question answering over the European Commission Joint Research Centre
technical guidance on Natech risk management, with the retrieval measured rather than read.

This is the code behind **Retrieval-Grounded QA over a Structured Hazard Guidance Corpus**,
Politecnico di Torino, DISEG, from October 2025. The system in `natech/store.py` and
`natech/answer.py` was written in April 2026 and is kept as it was. The corpus validation
and the retrieval evaluation were added in August 2026. Which
part is which is stated in every module docstring, and the April files are in `original/`
unchanged so the difference is checkable rather than asserted.

```
python3 demo.py                                # corpus, validation, retrieval sweep
python3 -m unittest discover -s tests          # 36 tests, nothing to install

pip install '.[rag]' && ollama pull mxbai-embed-large && ollama pull llama3.2
python3 -m natech.cli                          # the interactive system
```

The corpus layer and the retrieval evaluation need **nothing installed**. Only the vector
store and the answer chain need Ollama, and their imports are deferred so the rest of the
package works without it.

## The system

The guidance is segmented by chapter into 44 records with three fields: chapter number,
section title, body text. 107,312 characters, from a 243-character subsection to an
8,613-character one. Segmentation is at chapter granularity and the structure is kept
rather than flattened, so chapter and title travel with the text through embedding and
retrieval. An answer is then attributable to a named section, and a failure is diagnosable
as retrieval or as generation rather than as one undifferentiated mistake.

Embedding is `mxbai-embed-large` through Ollama into a persistent Chroma collection;
generation runs against `llama3.2` served locally. Local rather than through an API was
the original decision and it still holds: the corpus stays off third-party infrastructure
and the retrieval step is inspectable, which matters more than throughput at this stage.

Retrieval is exposed as a single object, so the generation side imports a retriever rather
than a vector store and the store can be replaced by a graph-backed or workflow-aware
retriever without the generation code changing.

## What August added, and why

### Retrieval you can measure

The April system had no evaluation. Quality was judged by reading the passages that came
back and deciding they looked relevant, which is the usual way and has two problems: it
does not survive a change to the chunking or the embedding model, and it cannot answer the
question actually in front of it, which is whether returning two passages is enough.

`natech/retrieval.py` measures recall at k against a 12-question probe set with the
chapters that should answer each one. Against the term-overlap baseline on the shipped
corpus:

| k | recall | MRR |
| --- | --- | --- |
| 1 | 33% | 0.33 |
| **2** (the April value) | **33%** | **0.33** |
| 3 | 58% | 0.42 |
| 5 | 58% | 0.42 |
| 8 | 67% | 0.43 |

The step from 2 to 3 is the finding. `k = 2` was a guess, and on this probe set it costs a
third of the questions the corpus can answer. Recall is still climbing at 8, the largest k
tried, so this sweep has not found where it levels off, and the report says that rather
than reporting the edge of the sweep as a plateau.

Two things this deliberately is not. It is not a benchmark: the probes are hand-written to
cover the document rather than sampled from real users, and real questions would be
harder. And it says nothing about whether the generated answer was any good, only about
whether the right section was available to generate from.

### A baseline the embedding retriever has to beat

The figures above are term overlap weighted by inverse document frequency, standard
library, no stemming and no phrase matching. It is here so the dense retriever has
something to beat, because a dense retriever that cannot beat counting shared words is not
paying for the model server it needs, and that comparison is not usually made because the
baseline is not usually built.

It also shows where it fails, which is informative. `What is a Natech accident?` is never
retrieved at any k: every content word in it is either a stopword or appears in most
sections, so term overlap has nothing to work with. That is precisely the question an
embedding retriever should win.

### Refusing to index a broken corpus

The April loader took whatever the CSV held. A row with an empty `Content` field became a
document with empty page content, which embeds to something meaningless, sits in the
index, and is retrievable as context for a question it says nothing about. Nothing raised
and nothing logged.

`natech/corpus.py` now blocks on an empty section, a duplicate chapter number (documents
are keyed on it, so a duplicate silently overwrites rather than adds) and a missing chapter
number. It flags for review, without blocking, a section below 120 characters, which is
usually a heading captured without its body, and a missing title, which costs attribution
rather than correctness.

### Rebuilding a partial index

The April version reused the store when the directory existed. That is right almost always
and wrong in the case that costs the most: a build interrupted part way leaves the
directory holding some of the corpus, and every run afterwards reuses it. The store then
answers from a subset of the guidance and nothing reports a problem. Reuse is now gated on
the store holding the expected number of documents, which is cheap to check.

### Citing the section

Chapter and title were already travelling with every passage and were not being used.
`answer()` now returns the sections it generated from alongside the text, so retrieval
failing and generation failing stop looking the same from outside.

## Limits

Read this before the code.

- **The probe set is 12 hand-written questions**, not a benchmark, and the expected
  chapters are my judgement of where each answer lives. It exists to make a change
  measurable, not to report a quality score.
- **The recall figures above are the keyword baseline**, because the embedding retriever
  needs a running Ollama and cannot be measured in CI. Reproducing the equivalent table
  for the dense retriever is one command locally and is the obvious next run.
- **Nothing here evaluates the generated answer.** Retrieval recall is a necessary
  condition for a good answer and nowhere near a sufficient one.
- **Chapter-level chunking is coarse** for the longer sections. The 8,613-character
  section is one unit, and a question about one paragraph of it retrieves all of it.
- **There is no reranking stage**, so raising k spends context linearly.
- **The corpus is a segmentation of a public JRC report** and the segmentation is mine. The
  report is cited below; nothing in this repository reproduces it beyond the extracted
  text used for retrieval.

## What I would do next, in order

1. Run the sweep against the embedding retriever and put both columns side by side. The
   machinery is there; it needs a machine with Ollama.
2. Replace the hand-written probes with questions from the people who would use this,
   which is the same argument for user studies I make in
   [geoqa-study-harness](https://github.com/daudee215/geoqa-study-harness).
3. Sub-chapter chunking with overlap for the long sections, measured against the same
   probes rather than assumed to be better.
4. A reranking stage, so k can rise without the context growing linearly.
5. Answer-level evaluation, which needs a gold answer per probe and is a much larger job
   than retrieval recall.

## Files

| file | what it holds |
| --- | --- |
| `natech/corpus.py` | loading, validation, and the refusal. August |
| `natech/store.py` | embedding and the persistent Chroma store. April, with the rebuild gate fixed |
| `natech/answer.py` | the prompt and the chain. April, with citations added |
| `natech/retrieval.py` | recall at k, the k sweep, the keyword baseline. August |
| `natech/cli.py` | the interactive loop. April |
| `original/` | the April files, unchanged, for comparison |
| `data/guidance.csv` | the 44 segmented chapters |
| `demo.py` | corpus, validation and the retrieval sweep, no dependencies |
| `tests/` | 36 unittest cases |

## Source

Krausmann, E. and Necci, A. *Natech risk management: Guidance for operators of hazardous
installations*. European Commission, Joint Research Centre. The report is a public
document. The segmentation into 44 chapter-level records is mine.

## Licence

MIT for the code. See `LICENCE`. The guidance text in `data/guidance.csv` is extracted from
the JRC report cited above and belongs to its publisher.
