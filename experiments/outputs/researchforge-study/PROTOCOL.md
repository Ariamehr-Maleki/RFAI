# ResearchForge: offline collaboration ranking study

Protocol drafted 2026-10-25, before any benchmark scores were observed.

Engineering update27October, before final-label access: equal identifier/category
strings are interned in historical loading. Full TRAIN output fingerprints,
including insertion order, match the preserved original loader; retained RSS fell
by508,583,936 bytes. Query/feature/model/scoring definitions did not change.
The interrupted29-query pilot was archived and identical query caches reused.
Supervisors now admit relevant optimized jobs at2500MiB available RAM and keep
640MiB runtime/checkpoint reserves. Final text encoding can overlap one
single-thread expanded worker after the complete TRAIN vector audit; its encoder
uses two threads. Such contended logs are not isolated cost measurements.
The final-cost job still waits for both development and encoding to finish.

Scheduling correction27October~12:30Tehran: the attempted overlap reached the
640MiB reserve and stopped expanded collection after117 committed queries.
All117 receipt hashes were verified; only the confirmed dead worker's stale
collection lock was removed. Final label-free encoding now finishes first, after
which expanded development resumes unchanged. The failed overlap is retained
as engineering history and supplies no isolated performance claim.

## Scope and claim boundary

Execution specification27 October2026, before final-label access: cost profiling
uses30 fixed development query IDs (lexicographically first saved validation
metadata files),3 timing repeats after a parity/warm-up pass, model seed42,
padding42 and1000 candidates for full/no-content pipelines. Retrieval and feature
construction are timed jointly; lexical/semantic similarity calls are subtimings,
not a complete causal decomposition of cost. Model scoring plus sorting is timed
separately.20ms sampled process RSS is not guaranteed OS peak. Competing study
workers must be absent; actual measured costs remain pending.

Final secondary subgroup code averages query-level method differences across
model seeds before joint author-cluster resampling. The existing1–3,4–10,11+
strata and new/repeat relevant sets are retained; groups with fewer than20 author
clusters are flagged as sparse. Secondary marginal intervals and direct
between-stratum effect differences do not replace the frozen primary Holm family.
The actual comparison pairs and sampling sizes still require the global lock.

Preprocessing extension26 October14:40 Tehran: the separate final-corpus
lexical index is prepared without relevance labels, reading only corpus_id,
date,title,abstract. Dates support the same strict historical mask as TRAIN;
roles/authors/references/judgments are not read. Stateless hashing parameters
are unchanged, with no vocabulary/IDF fitting or hyperparameter selection.
Row order must match the prior label-free semantic reuse inventory and source
checksum. This is preprocessing, not target selection or final evaluation.

Development execution extension,26 October10:25 Tehran: expanded PreScience
uses a fixed1200-query pre-July training pool (sampling seed20261026),1000
training groups per model seed42/43/44, and500 common post-July validation
queries (sampling seed20261027). Candidate padding seed42 remains common.
Full and no-content modes share the query manifest. No-content removes
lexical/semantic retrieval and all seven content feature columns, including
category cosine; it retrains each learned model. The eight fixed-pool factorial
fits are separate from this end-to-end ablation. Both learned tree families
retain three declared configurations; logistic has three C values. All
variants and negative results are preserved. This is adaptive development,
not a final-test sample-size lock. Five exploratory summary contrasts use
seed-averaged query effects and author-cluster inference; their Holm family
is explicitly separate from the eight planned final primary contrasts and
does not correct validation model-selection bias. The actual expanded run
waits for completed pilot/NCN dependencies and resource gates.

User-selected target as of26 October2026: Scientific Reports. The dated
`SCIENTIFIC_REPORTS_PLAN.md` extends this protocol before held-out test access,
after observing development outcomes. It defines the eight-cell primitive-feature
factorial, explicit age/decay interpretation, end-to-end content ablation,
subgroups, primary comparisons and isolated cost study. Read it with this file;
it does not create a final configuration lock or imply preregistration of earlier
development. Three structural tests validate the component mapping; empirical
factorial fitting and final-run integration are still pending.

No human evaluation, surveys, interviews, LLM-as-judge substitute, or invented annotations.
Evaluate ranking of observed collaborations and a separate expert-retrieval task. Historical coauthorship is an observed positive, not proof of compatibility; unobserved pairs are unlabeled negatives. Do not claim user satisfaction, causal productivity improvement, or quality of generated proposals.

Hypothesis: combining temporal network evidence and publication-content similarity with a library-based LambdaMART ranker improves retrieval over frequency, semantic retrieval, and fixed fusion, with particular attention to first-time collaborations. This is a hypothesis, not an established novel algorithm or promised positive result.

## Datasets and units

| Dataset | Unit and task | Main metrics | Purpose |
|---|---|---|---|
| [PreScience](https://huggingface.co/datasets/allenai/prescience) | Seed author, date -> other authors of future paper | nDCG@1000, R-precision; nDCG@10, Recall@10/50, candidate recall | Primary collaboration benchmark |
| [ogbl-collab](https://ogb.stanford.edu/docs/linkprop/) | Historical author graph -> future collaboration edges | Official Hits@50 against shared 100,000 negatives | Independent temporal graph benchmark |
| [exHarmony](https://huggingface.co/datasets/Reviewerly/exHarmony) | Paper description -> experts | Official qrel-aware ranking metrics after inspecting release | Secondary expertise retrieval; never described as collaboration outcomes |

PreScience and OGB: ODC-BY. exHarmony: CC-BY-4.0. Preserve attribution, checksums, exact source revisions and downloaded manifests. Kaggle arXiv is not an independent labeled matching benchmark and will not be added merely to increase dataset count.

## Splits and leakage prevention

1. Keep PreScience official test period sealed during development. Use chronological subdivision of official training target papers: before 2024-07-01 training, from 2024-07-01 validation. Inspect actual dates and report counts. All historical records must have date strictly before query date; same-day records excluded. Only rows with role `target` become supervised queries. Seed is the first eligible author with prior publications, matching official dataset creation.
2. Fit preprocessing and learned models on training only. A stateless text hashing transform is allowed because it does not learn from future text. Pretrained model dates and potential contamination must be recorded.
3. No target title/abstract, future collaborators, future citations, or target-author h-index enters collaborator prediction. Recompute graph features at each cutoff. Labels never select validation/test candidates; report missed-positive candidate recall. Training-only positive injection, if used, is explicitly logged.
4. OGB: <=2017 train, 2018 validation, 2019 test; graph feature training must use a pre-2017 snapshot when 2017 edges are training labels. No edge can serve as its own input feature. Start with fixed train graph for final comparison; a separate explicitly labeled variant may add 2018 edges after tuning. Preserve official negatives and evaluator tie semantics.
5. exHarmony: keep query documents and prohibited query-derived evidence out of author profiles. Use released judgments, expose proxy-label construction and label circularity. Determine its official split from code before committing training protocol. If only a test set exists, use author/query-disjoint development subset and clearly distinguish adapted evaluation from published benchmark.

## Baselines and experimental controls

- Random/popularity; coauthor frequency (official implementation on PreScience).
- Common Neighbors, Adamic-Adar, Resource Allocation on temporal graph.
- Lexical retrieval and semantic embeddings; published embedding-fusion and rank-fusion baselines where feasible.
- ResearchForge original scoring: retain documented weights/behavior, mark unavailable dimensions, distinguish exact reproduction from data-adapted variant. Original fabricated profiles cannot be used for evaluation.
- Fixed reciprocal-rank fusion / tuned weighted fusion.
- Logistic regression and LightGBM binary classifier on the same features to test whether ranking objective matters.
- `lightgbm.LGBMRanker(objective='lambdarank')`; a second library ranker is optional after the core comparison is complete.
- A recent strong graph baseline with public implementation on OGB; literature tables alone are not reproduction.
- **Required update from latest PreScience v2 (2026-07-02):** compare against its Autoresearch-discovered fixed pipeline, not only v1 Frequency. It uses 45-paper frequency, damped 2/3-hop traversal, citation-history author pull, and activity decay. Reimplement the described final scorer if source is absent, with all ambiguities disclosed; reproducing the expensive agent search is not needed to run that fixed scorer. Latest paper reports 0.49 nDCG@1000 versus 0.40 frequency; these are published figures, not our runs. Do not claim network fusion alone is new.

Primary comparison uses identical candidate pools and feature availability. Also evaluate the complete retrieval+ranking pipeline and standalone retrievers, so candidate generation benefits are not misattributed to ranking. Use the same tuning budget per learned method and the same validation queries. Choose configurations only on validation. Freeze configuration and hashes before one final test run. Never combine scores from different tasks into a single accuracy.

## Required analyses

At least 3 seeds for stochastic methods (5 preferred after runtime measurement), paired seed-author cluster bootstrap confidence intervals on PreScience; graph dependence must be acknowledged for OGB. Paired tests on the same queries, Holm correction for declared primary comparisons. Report effect sizes and all outcomes, including failures. Ablate temporal, network, text and learning-to-rank components. Report first-time vs repeat collaboration, 1-3 prior papers vs longer history, and performance/cost (wall time, memory, model size). First-time label sets and candidate exclusions must be specified consistently.

No test-set tuning. Pilots use training-period validation data only and are labeled developmental, not paper results. An initial graph-only pilot is a pipeline check, not the final ResearchForge method.

## Deliverables and gates

1. Data manifests, quality and leakage report; reproducible commands and pinned environment.
2. Working pilot with saved predictions, metric unit checks and protocol checks.
3. Full baselines, tuning log, locked configuration, seed runs, ablations and uncertainty.
4. Results CSV/JSON, publication figures, paper draft with verified references and honest limitations.
5. Submission-ready package after authorship, affiliations, funding/conflicts and final journal selection are supplied by authors. No submission or fee payment is authorized by this research task.

The requested few days is an execution target, not a guarantee of statistically favorable results, journal acceptance, or publication within days. If gains fail to replicate, report them and reconsider the contribution on validation data.

## Source versions

- ResearchForge: `57243e190904f6ca7e6796734c32878ba8cd6c45`.
- PreScience official code: `47384ed79c48cfef610f30af12ba883f2a0093cb`.
- [PreScience latest paper v2](https://arxiv.org/html/2602.20459v2), especially Section 4.2 and Appendix C.2. Earlier v1 was also inspected. The downloaded train release currently has 373,716 rows while v2 describes 373,707; record and resolve this version discrepancy before claiming exact paper reproduction.
- [LightGBM ranker documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html).

## Development extension, 25 October, before text/graph scores

The next run uses 200 training and 100 validation queries, 1,000 candidates, seed42.
The validation sample is the same deterministic sample as the first pilot. Three
LightGBM configurations per family remain (7 leaves/60 trees,15/100,31/100).
Selection now follows the stated primary metric nDCG@1000, rather than pilot nDCG@10.
This is a development change; all final official test labels remain sealed.

Text is a stateless 262,144-dimensional nonnegative hashed unigram TF representation,
English stop words removed, each document L2-normalized. This is a **lexical**
baseline, not BM25, TF-IDF, or a pretrained semantic model. The query is the
normalized mean of the seed's last10 eligible document vectors. Retrieve top2000
historical papers, sum their cosine scores by author. Future/same-day rows are
masked before retrieval; neither vocabulary nor IDF is learned from the full file.

The shared pool uses reciprocal-rank fusion (constant60) of lexical retrieval,
recent10 frequency, and the description-based Autoresearch implementation. Primary
reranker ablations keep this shared pool fixed; thus removing text from the
reranker does NOT remove text from candidate generation. Standalone retrievers
are reported separately. Training-positive injection is logged, never applied to
validation/test. Exact candidate counts and candidate recall accompany each run.

Autoresearch v2 does not specify every numerical operation in its paragraph.
The executable ASSUMPTIONS in prescience_text_graph.py records the chosen path
normalization, citation boost, compound third-hop damping, and tie handling.
Call this an approximate description-based comparator throughout. Do not present
its score as exact reproduction of the published baseline. Add sensitivity or
official code verification before treating it as the definitive strong baseline.

For exHarmony, construct an **adapted**60/20/20 split using whole components of
queries sharing any query author or normalized title; assign components without
model scores. Query authors are disjoint, but citation-qrel relevant authors can
overlap. Two records displayed during format inspection are quarantined to train
with their entire components. Reject the split if any held-out partition has
fewer than500 queries. Remove any collection document matching a query ID or
normalized title across all partitions. The corpus lacks dates, so independently
verified per-paper temporal ordering cannot be claimed. Do not use undated author
citation totals, experience, or works_count features. Freeze IDs and checksums
before reading relevance labels for model development. Official PreScience and
OGB test labels remain unused.

## Development extension, 25 October, semantic and OGB expansion

MiniLM encoding uses `sentence-transformers/all-MiniLM-L6-v2`, pinned revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, title plus abstract, 256-token
truncation, attention-mask mean pooling and L2 normalization. Inference is local
CPU; float16 storage has 384 dimensions. This is a general semantic comparator,
not a scholarly encoder or evidence of absence of pretraining contamination.
Only TRAIN papers are encoded. Retrieval requires a completed index and filters
the seed history and candidates to dates strictly before each query. The
pretraining corpus includes scientific data; model/data temporal provenance must
remain disclosed. A partial encoded corpus must not be used as a full baseline.

OGB expanded development uses 10,000 sampled 2017 positive edges per seed
(42,43,44), five sampled negatives per positive, graph context through2016 for
training and through2017 for official2018 validation. Each LambdaMART and
classifier family receives the same three leaf/tree configurations as earlier.
The primary track remains graph-only. A supplementary track adds cosine,
Euclidean distance,128 coordinate products and128 absolute differences of the
released node features; their historical construction cutoff is unverified.
Source-cluster percentile bootstrap intervals use2000 draws conditional on the
fixed100,000-negative pool. They do not capture all crossed network dependence
and are exploratory because validation also selects configurations. Across-seed
variation includes sampling changes, not solely optimization randomness. Test
labels remain sealed; this extension is not a final configuration lock.

### TRAIN integrity correction,25 October, before subsequent PreScience runs

A full TRAIN audit found3,147 papers with duplicate nonempty author IDs. Future
experiments deduplicate author IDs within each paper, retaining first-occurrence
order so the seed author order is preserved. This avoids counting one person's
appearance multiple times in frequency, citation and lexical-author aggregation.
Previously saved pilots used raw author lists and remain preserved as preliminary
artifacts. Corrected runs use new `pilot-dedup-*`/`text-graph-dedup-*` directories;
the latter also hashes the data-loader source into its configuration. The pinned
official frequency code iterates raw author entries, so report this cleaning as
a deviation and retain a raw-input official-baseline sensitivity when checking
parity. Do not silently treat older and corrected runs as the same protocol.

All373,716TRAIN corpus IDs are unique; the9-row difference from v2's373,707 is
therefore not explained by duplicate corpus IDs. There are66 normalized-title
repeat groups and67,671 records without nonempty author IDs. All target dates
fall in the declared window. Of669,754 reference edges,18 have a later referenced
paper date and2 have the same date; strict historical masking remains required.
No evidence here proves that the title repeats are duplicates or that citation
date anomalies are errors rather than preprint/version effects.

## Development extension,25 October18:25, before hybrid/BM25 scores

`prescience_hybrid_graph.py` adds MiniLM paper retrieval as a fourth RRF channel
alongside lexical, frequency10 and approximate Autoresearch. It adds three
semantic features (mean/max recent-paper cosine, retrieved-paper author sum),
giving23 total. Same1000-candidate joint pool for all learned methods; separate
standalone lexical/semantic/network retrieval remains reported. Ablations remove
semantic, lexical, all text, or recency features and fused scores that could carry
the ablated evidence. Removing reranker features does not remove their influence
on the fixed candidate pool. The full semantic index is mandatory; partial
embedding coverage is rejected. Begin with the same200/100 sample before scaling.

The complementary exHarmony extension uses hashed-term BM25 with k1=1.2,b=.75,
binary query-term occurrence and positive IDF log(1+(N-df+.5)/(df+.5)). Defaults
and IDF reference: [Lucene BM25Similarity](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html).
It is not exact Lucene: existing262144-bin hashing collisions remain, lengths
are not quantized, and all filtered candidate documents contribute to corpus
statistics. Re-tokenize the fixed corpus and verify every stored normalized
posting reconstructs the exact integer hashed count before use. No relevance
labels enter corpus statistics. Same200/100 sampled queries,1000-candidate cap,
Authors training labels,3 matching LightGBM configurations and frozen split as
the prior cosine pilot. Cite/SimCite remain secondary transfer evaluations.
Metrics keep the original missing-author judgments and are independently checked
with pytrec_eval. No final test lock or superiority claim follows from this pilot.

## Development extension,25 October, larger exHarmony and original scorer

Before inspecting expanded scores, fix exHarmony sample sizes to1000 training
and500 validation queries per seed42/43/44,1000 candidates, unchanged BM25 and
three matched LightGBM configurations. Seeds change sampled queries as well as
training randomness; overlap is possible and sample SD is not a confidence
interval. Independently verify every saved metric with pytrec_eval. Continue
to preserve all raw relevance judgments, including unavailable authors.

The original ResearchForge score is ported numerically and checked against the
AST-extracted original function on25 fixture cases. The PreScience adapter uses
unique categories from the last10 strictly prior papers as research-interest
proxies. Skills are unavailable and empty; geography uses the original default50,
career85 and missing-semantic50 are retained. Original weights and rounding are
unchanged. Score the existing corrected100-query validation shared pool; this
is a data-adapted legacy comparator, not the original live system's end-to-end
performance and not a sufficient primary baseline. Do not infer that missing
metadata proves the original product has poor real-world matching quality.

## Development extension,25 October20:05, NCN and fixed-scorer sensitivity

After a1000-positive one-epoch resource pilot, commit NCN development to10000
sampled2017 positive edges,5 fixed negatives per positive, seeds42/43/44 and100
epochs. Reuse the unmodified pinned upstream GCN/CNLinkPredictor architecture.
The positive/negative sampling function and graph cutoffs match the expanded
tree comparisons. Input is the released128-dimensional node features, with the
same provenance caveat as the supplementary feature-based tree track. Select
the highest validation Hits50 checkpoint; disclose100 epoch comparisons versus
the3-configuration tree selection budget. Save optimizer and RNG state for
epoch-boundary resumability. This adapts the training protocol and is not a
reproduction of the published NCN leaderboard number. Official test stays sealed.

Before viewing sensitivity scores, define six interpretations of the approximate
PreScience v2 fixed scorer: current default; citation weight0,.25,4 versus1;
independent third-hop damping.3 rather than compounded.6*.3 (implemented as.5
times the already damped second hop); and unnormalized outgoing transitions.
Everything else remains fixed. Report every variant on the same existing100
development queries, both shared candidate pool and unpadded standalone top1000.
Default numerical parity with saved features is mandatory. No best-variant
selection is confirmatory, and none is called exact official Autoresearch code.

## Final-run implementation guard (prepared, not activated)

Prepared inference implementation,26 October: for a common set of query IDs,
average paired per-query metric differences over the three model seeds before
seed-author cluster resampling. Use10000 bootstrap draws for marginal95% intervals
and10000 random cluster sign flips for two-sided p-values (Monte Carlo plus-one
correction); Holm adjusts the predeclared family. Sign flips assume symmetry of
paired cluster effects under the null and do not eliminate crossed network
dependence. Relevant sets, source authors and eligibility must match across
methods and model seeds. Pure helper tests pass; no final inference executed.

`final_evaluation_gate.py` is an audit mechanism for the eventual final runners.
It requires explicit configurations for all three benchmarks and declared
method comparisons, seeds, uncertainty and Holm policies. It hashes required
code/model/data/split/protocol/environment/completed development-report files,
creates a new lock without overwrite, and checks those hashes before recording
test authorization. A different second run ID is rejected; the identical frozen
run ID can resume interrupted computation. Final runners must consume the
returned frozen configuration and invoke the guard before opening held-out
labels. The implementation is not an OS security boundary or evidence of a
methodologically adequate plan on its own. Six synthetic guard fixtures passed.
No real configuration has been frozen and no test run has been authorized.

The user subsequently revoked the former28 October stopping date. Continue
until actual deliverable completion. Only scheduling guards change; scientific
settings, sealed-test policy and experiment provenance are preserved. Archived
original sources and an audited source-hash migration support NCN resumption.

## Label-free TEST companion-corpus planning,25 October23:10

The official PreScience TEST companion corpus contains464942 papers, compared
with373716 in TRAIN. Do not silently restrict final TEST candidate/history
coverage to the TRAIN corpus. A preprocessing inventory reads only corpus_id,
title and abstract; no author, role, reference, citation or relevance fields.
Reuse an embedding only when the corpus ID and SHA256 of the exact encoder input
match. This identifies336792 reusable rows and128150 rows requiring encoding;
no shared-ID texts changed. Reuse also requires a completed TRAIN manifest and
identical pinned model revision, tokenizer, truncation, pooling and normalization.
The plan records file hashes and requires98,419,200 additional vector bytes, or
357,075,456 bytes for a fully materialized TEST index. It is not a configuration
lock or authorization to read labels. Future scoring must apply strict historical
date masks and exclude target text, and must pass the final evaluation guard.

The prepared TEST text-cache builder verifies the completed TRAIN-vector file,
model files, input source and reuse-plan hashes before reuse. It reads only the
three allowed text/ID columns, writes resumable atomic checkpoints and rejects
corrupt committed prefixes or concurrent writers. Synthetic integration tests
cover interruption and label-column exclusion. Actual TEST encoding has not yet
started. Its eventual completion alone will not authorize held-out evaluation.
