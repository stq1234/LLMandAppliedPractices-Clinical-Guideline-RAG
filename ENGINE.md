# Clinical evidence QA engine

This 7.3 system answers general questions about adult blood-pressure measurement
and diagnosis from the approved data/evidence.jsonl corpus. It does not use
evaluation questions or labels. A response is composed from evidence-backed
claims, with each returned citation resolving to its source, authority, version,
population, locator and URL. Unknown IDs, unsupported claims and unverifiable
quote anchors are rejected.

## Run

From the project root, after model_backend.py and the approved evidence corpus
are present:

~~~sh
python cli.py --query "How should blood pressure be measured and confirmed for adults?" --mode adaptive
~~~

The command returns JSON with the answer, citations, source records, routing
trace, initial and final evidence IDs, retrieval-round count and diagnostics.
--corpus and --index override the default paths. --params params.json loads
routing settings, including those selected from historical development logs,
through the normal CLI entry point.

An optional --supplied ID1,ID2 passes preloaded evidence by corpus ID only.
reuse uses only those IDs and performs zero retrieval rounds; single performs
one retrieval round; multi plans evidence needs and performs two retrieval
rounds; adaptive selects reuse, one round or two rounds from query-visible
features and may add a second round when first-pass evidence checks reveal a
sufficient gap. adaptive_nocrit uses the adaptive router without the semantic
evidence checker, while retaining evidence-ID and exact quote-anchor
validation. The checker is an inference-time, Self-RAG-inspired step; it is not
a model trained with the original Self-RAG objective.

Routing defaults are initial_threshold=0.35, supplement_threshold=0.25,
multi_threshold=0.50, initial_k=5, supplement_k=5, max_evidence=8 and
lexical_weight=0.15. force_action (reuse, single or multi) is available in
adaptive modes for replaying a historical route. initial_threshold controls
reuse of supplied evidence by lexical coverage; multi_threshold selects the
initial multi-round route from the aspect score; supplement_threshold controls
adaptive escalation after first-pass support checking. The numeric feature
vector is available as System.features(query, supplied=None).

The public API is:

~~~python
from model_backend import Backend
from clinical import System

system = System(Backend(), corpus_path="data/evidence.jsonl", index_dir="index")
result = system.respond(
    "How should blood pressure be measured and confirmed for adults?",
    mode="adaptive",
)
~~~

Corpus records have id, source_id, authority, title, version, locator, url,
kind (guideline or review), population, text and topics. New corpora may also
provide population_scope with explicit included and excluded labels, such as
included adult and excluded pregnancy. Legacy population prose is interpreted
adult-first: an adult guideline remains adult evidence when its scope paragraph
notes that pediatric or pregnancy care is addressed elsewhere. Such exceptions
do not reclassify the entire guideline. The local vector cache is keyed by the
backend embedding signature, record ID and embedded-content hash. Changed
records are re-embedded; removed records leave the current index.

For non-identifying development queries, --audit-log audit.jsonl appends the
redacted query, its digest, route, evidence references, UTC time and system
version. Avoid putting patient names or other identifiers in queries; routine
responses do not write logs.
