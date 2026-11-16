"""One documented editorial migration; no experimental computation."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];path=ROOT/'outputs/researchforge-study/MANUSCRIPT_DRAFT.md'
text=path.read_text(encoding='utf-8')
a=text.index('## Abstract');b=text.index('## 1. Introduction')
text=text[:a]+'''## Abstract — awaiting held-out results

Scientific collaborator recommendations should be evaluated against the information available at prediction time and distinguished from recommendations for established collaborators. We study a reproducible retrieval-and-ranking pipeline combining publication content, historical collaboration networks and explicit recency features. A library implementation of LambdaMART is compared with matched classifiers, fixed scoring and retrieval baselines. Eight feature combinations, an end-to-end content ablation, history strata and new-versus-repeat collaboration metrics separate the sources of predictive performance. Temporal graph prediction and an adapted expertise-retrieval benchmark provide complementary tasks. The study also measures warm pipeline latency and reports selection budgets, missing-profile relevance and provenance limitations. Held-out effect sizes and the final conclusion will be inserted only after the globally frozen evaluations complete. Existing validation findings are mixed and do not establish a general ranking-objective advantage.

Keywords: scientific collaboration; learning to rank; temporal evaluation; expert retrieval; reproducibility

'''+text[b:]
a=text.index('## 4. Implemented method');b=text.index('## 7. Reproducibility')
text=text[:a]+'''## 4. Methods

### 4.1 Historical content retrieval

For collaborator forecasting, a seed author is represented by up to ten publications strictly earlier than the query date. Lexical vectors use stateless nonnegative unigram hashing into262,144 dimensions with English stop-word filtering and L2 normalization. No vocabulary or IDF is fitted. Semantic vectors use the pinned `sentence-transformers/all-MiniLM-L6-v2` checkpoint, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with title and abstract concatenation,256-token truncation, masked mean pooling and L2 normalization. Vectors are stored as float16 and renormalized during blockwise cosine scoring. The checkpoint is not fine-tuned on any study partition. Its pretraining provenance is a separate limitation from our publication-date masks; the experiment does not prove absence of pretraining overlap.

Each content channel averages eligible seed-paper vectors, scores the historical corpus and retains at most2,000 positively scored papers. Paper evidence is aggregated to authors. Reciprocal-rank fusion with constant60 combines lexical, semantic, recent-collaboration frequency and the description-based fixed scorer. The shared pool contains up to1,000 eligible authors, with deterministic historical-author padding when necessary. Candidate IDs are sorted before fitting/scoring, making stable score ties equivalent to ascending string author IDs. Standalone channel rankings are evaluated separately. Training may append missing relevant candidates, with injection counts retained; validation and final evaluation never inject positives.

### 4.2 Features, training and component comparisons

The practical model uses23 features. Three history controls record seed and candidate publication counts and their balance. Seven content features describe category cosine and lexical/semantic publication and retrieval scores. Nine network features describe recent and lifetime collaboration counts, neighborhood/degree structure, reverse collaboration frequency and citation-history author counts. Two explicit-time features measure last-collaboration recency and candidate activity decay. The remaining two features are the composite fixed score and reciprocal-rank fusion score. The ordered feature schema and every fitted model are stored with checksums.

LambdaMART uses LightGBM's `LGBMRanker(objective="lambdarank")` with one group per query [7]. The matched `LGBMClassifier` receives identical candidate rows and features. Each tree family evaluates the same three leaf/tree configurations:7/60,15/100 and31/100, with learning rate0.05 and L2 regularization1. A standardized logistic regression control evaluates C in0.1,1,10. Configuration selection maximizes development nDCG@1000, with the first declared configuration breaking exact ties. Saved models are replayed to verify export equivalence.

Expanded PreScience development uses a fixed pool of1,200 pre-July2024 queries,1,000 training groups per model seed42,43,44, and500 common post-July validation queries. The two candidate modes and all component comparisons share query identities and a padding seed of42. Seed variation reflects model/training-sample changes, not independent validation populations.

Eight factorial models retain the three history controls and add every combination of content, network and explicit-time features. Both composite scores are excluded from every factorial cell, including the full21-feature cell, preventing those aggregates from reintroducing removed signals. This factorial holds the full candidate pool fixed. A separate no-content pipeline removes lexical/semantic candidate generation, zeros and excludes all seven content features, and retrains each learner. Its candidate recall and ranking performance must be interpreted together. Strict historical eligibility and recent10/45-paper windows remain in every cell: removing explicit-time columns does not remove all temporal information.

### 4.3 Fixed and complementary comparators

The PreScience v2 comparator is a description-based reimplementation, not an exact official Autoresearch reproduction [3]. It combines45-paper frequency, weighted multi-hop coauthor propagation, recently cited-paper authors and activity decay. Transition normalization, citation weights and the interpretation of third-hop damping require implementation assumptions. Six prespecified development variants preserve these ambiguities; the strongest shared-pool validation variant is selected before final-label access. An audit of the available official frequency helper verifies only that component. It does not validate the whole composite scorer.

OGB development samples10,000 positive2017 edges per seed with five sampled negatives per positive, using features from the graph through2016. Validation uses the graph through2017 and the official2018 positives and shared negatives. Eight graph features and released node-feature extensions are compared with fixed graph scores and Neural Common Neighbor. Each tree family has three parameter configurations. NCN is trained for100epochs per seed and selects a saved checkpoint by validation Hits@50; this larger checkpoint-selection budget is explicitly unequal. Final inference retains the graph through2017, without adding validation edges. The temporal provenance of the released node vectors is not certified, so their results are an additional track.

The exHarmony adaptation retrieves papers using hashed BM25 with k1=1.2,b=0.75 and binary query term frequency, then aggregates candidates and seven features to author profiles. Hashing is not exact Lucene/Pyserini BM25. Authors judgments train and select the models; Cite and SimCite assess descriptive label transfer. Missing-profile relevant authors remain in the original denominators, and empty retrieval outputs remain in evaluation. The held-out author/title-component split is shared across all model seeds. This is an adapted expertise task, not evidence about actual collaboration choices or reviewer willingness.

### 4.4 Metrics and uncertainty

PreScience's primary metric is nDCG@1000, with nDCG@10,R-precision,Recall@10/50 and candidate recall as secondary measures. OGB uses official strict-threshold Hits@50 against the shared negative pool. Adapted exHarmony uses Authors nDCG@10 as primary; Cite/SimCite are secondary. No metric is pooled across these different tasks. Independent evaluators verify saved predictions and aggregate metrics.

Before any final relevance is read, one global manifest must bind all three dataset configurations, code, model artifacts, feature schemas, candidate policies, query sampling, comparisons and input checksums. Final runners cannot fit or tune models. A deterministic final PreScience sample size or census is selected from development runtime, available storage and inferential precision, not from test outcomes; the frozen manifest records that choice. The main results must identify any subset explicitly.

Paired query/edge effects are averaged across model seeds before10,000 cluster-bootstrap draws and cluster sign-flip tests. Clusters are the seed author for PreScience, source node for OGB and adapted query author/title component for exHarmony. Marginal95% intervals are reported alongside separate Holm-adjusted primary comparison families for each task. Three-seed standard deviations are descriptive and are not confidence intervals. Shared neighbors, target authors or cited experts can create crossed dependence that these clusters do not fully capture.

PreScience's eight primary contrasts compare practical LambdaMART with its matched classifier, frequency45, semantic retrieval, RRF and the development-selected fixed scorer, and compare the full factorial cell with its three leave-one-component-out cells. Secondary analyses use prior-history groups1–3,4–10 and11+, plus new and repeat relevant collaborators. Both recall measures use the unchanged full ranking; empty relevant subsets are excluded only for that metric. Joint resampling estimates differences of subgroup effects directly. Groups below20 author clusters are flagged as sparse. These observational comparisons do not establish causal mechanisms.

### 4.5 Computational cost and implementation

An isolated cost run uses30 fixed development query IDs, three timing repeats after a parity/warm-up pass, seed42 models and1,000 candidates. It measures joint retrieval and feature construction, lexical/semantic similarity subtimings, and model scoring plus sorting. Similarity subtimings omit aggregation and do not isolate a pure network cost. Warm query latency is reported separately from preprocessing and training; missing active-time accounting is disclosed rather than inferred from interrupted wall-clock logs. Process RSS is sampled every20ms and is not a guaranteed OS peak. Competing study workers are excluded, although desktop processes can still cause contention.

Execution uses a four-logical-CPU desktop with12.77GB physical RAM. Indices use memory-mapped arrays. Exact-value string interning reduced retained TRAIN-loader memory by508.6MB; fingerprints of every paper, history and date entry, including insertion order, were identical before and after this engineering change. Query-level checkpoints and SHA256 receipts support interrupted resumption. A separate NCN runtime uses its compatible Torch/sparse-extension versions; package versions and source revisions are recorded.

### 4.6 Use of AI-assisted software

OpenAI Codex assisted literature retrieval, code development, local experiment execution, debugging, analysis and manuscript drafting. Software tests and independent metric checks are recorded. Codex is not an author. The human authors must review the complete evidence, approve the final text and take responsibility for its accuracy before submission; this statement does not assert that such review has already occurred.

## 5. Results — held-out analysis pending

No final-test result is reported in this draft. The three globally frozen evaluations, final component effects and isolated cost measurements are still incomplete. Final tables and figures are generated only after report identities, configurations and output checksums pass verification.

Development evidence is retained separately in `DEVELOPMENT_MANUSCRIPT_HISTORY_20261027.md` and the run reports. Some early pilots used duplicate author entries; their outcomes must not be substituted for corrected results. The completed expanded OGB validation mean Hits@50 is0.630617 for graph LambdaMART,0.633352 for its matched classifier and0.639638 for NCN, with sample standard deviations0.002872,0.006332 and0.002846. These are validation-selected results with unequal NCN/tree selection budgets.

Across three adapted exHarmony development seeds, mean Authors nDCG@10 is0.123936 for maximum BM25,0.118948 for LambdaMART and0.114370 for the classifier. Citation-derived judgments give a different pattern, but do not establish an Authors advantage. Expanded semantic PreScience development is in progress. These mixed findings motivate precise final comparisons and preclude a general superiority claim at this stage.

## 6. Discussion — interpretation boundaries

The intended contribution is evidence about when existing ranking tools help under a reproducible historical protocol. Combining text, networks and time is already studied; neither that combination nor using an existing learning-to-rank library is claimed as a new algorithm. A useful conclusion may be conditional or negative. Gains on shallow new-collaborator recall, aggregate ranking and candidate coverage are distinct outcomes and must not be substituted for one another.

Observed coauthorship provides positive retrospective evidence, while unobserved pairs are not confirmed unsuitable partners. The study does not measure willingness, availability, satisfaction, productivity, invitation acceptance or generated-proposal quality. It uses no human evaluation or LLM judge. Retrospective author disambiguation, corpus construction, incomplete expert profiles and unknown embedding pretraining overlap limit interpretation. The unresolved PreScience training-row discrepancy, approximate fixed scorer, adapted exHarmony split and node-vector provenance prevent claims of exact published-benchmark reproduction. Query subsampling, limited model seeds and incomplete crossed-dependence modeling must accompany the final uncertainty estimates.

The final discussion must tie each substantive claim to its frozen effect estimate, interval, candidate-coverage result and measured cost. Until those outputs exist, the draft cannot conclude where content or explicit temporal information is beneficial.

'''+text[b:]
text=text.replace('## 7. Reproducibility and availability','## 7. Reproducibility and availability')
path.write_text(text,encoding='utf-8')
print('Updated draft; historical manuscript preserved; no final-result claims added.')
