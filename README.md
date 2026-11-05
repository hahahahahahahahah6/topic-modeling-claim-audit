# Documentation claim auditor

Version 0.1.0. An offline, deterministic, standard-library-only CLI for checking
latent theme discovery claims against Python source. No network, telemetry,
LLM, or output file writes. Requires Python 3.9 or newer.

## Installation and use

```sh
python3 -m pip install .
topic-modeling-claim-audit --version
topic-modeling-claim-audit audit .
python3 -m topic_modeling_claim_audit audit . --json --verbose
```

The directory defaults to the current directory. JSON output is an array of
findings, each with verdict and message fields. Verbose decisions go to standard
error. Repeat the ignore option to skip relative path prefixes; multiple prefixes
per option are accepted. The failure threshold defaults to high. Exit status is
0 for no findings at the threshold, 2 for findings at the threshold, and 1 for
usage or input/output errors. A medium finding can therefore coexist with exit 0.

```sh
topic-modeling-claim-audit audit . --ignore vendor/ --ignore fixtures/ --fail-on medium
```

## Scan behavior

The root README and Markdown beneath the documentation directory are checked.
Markdown fixtures beneath test directories receive medium severity; shipped
claims receive high severity. Fenced code is excluded; inline code still counts.
Negative claims are removed before positive claims are checked. Matching is
case insensitive, exact substring, longest first with matched spans blanked.
Hyphens are preserved. A denial without source evidence passes.

```text
README.md
docs/**/*.md
tests/**/*.md
documented-topic-modeling-claim-false
documented-no-topic-modeling-claim-false
```

Python files are parsed without execution, excluding version control, bytecode,
and virtual environments. Construction and method calls must occur in one
module. Import aliases and assigned or chained instances are recognized; local
function and class definitions shadow imported names. Supported model pairs and
24 generic method names provide evidence. Bare imports do not.

```text
gensim.models.LdaModel / LdaMulticore: get_topics, show_topics, print_topics, get_document_topics
sklearn.decomposition.LatentDirichletAllocation / NMF: fit, fit_transform
bertopic.BERTopic: fit_transform, fit
gensim.models.wrappers.LdaMallet: show_topics, get_topics, print_topics
gensim.models.Nmf: get_topics, show_topics
```

## Differentiation

This auditor checks latent theme discovery. The following neighboring tools
check salient-term harvesting, passage condensing, text splitting, suffix
stripping, dictionary base forms, and label assignment respectively. Those
operations alone do not support a latent theme discovery claim.

```text
keyword-extraction-claim-audit
summarization-claim-audit
tokenization-claim-audit
stemming-claim-audit
lemmatization-claim-audit
classification-claim-audit
```

## Limitations

This is static pattern matching, not semantic judgement or proof of execution.
Exact phrases are recognized; paraphrases are missed. Imports are not executed,
and dynamic dispatch or cross-module instance tracking is not supported.
Syntactically invalid source is skipped as evidence. A matching generic call on
an unrelated object counts because its method name is shared:

```python
unrelated.extract_topics()
```

Substring matching can produce high severity false positives for unrelated
words. These examples match the short bare phrase despite their different
meanings; this known collision is deliberately preserved:

```text
topic
topics
topical antibiotics
subtopic headings
subtopics
```

Clustering on TF-IDF vectors is deliberately not counted, although some
practitioners use it for latent theme discovery. Salient-term extraction,
summarization, latent semantic analysis, embeddings, sentiment analysis, entity
recognition, grammatical tagging, text splitting, suffix stripping, and base-form
conversion alone do not qualify. Subprocess invocations do not qualify.

## Development

```sh
python3 -m unittest discover
python3 -m build
```

Licensed under MIT.
