# Final evaluation runner contract

Status27 October2026: all three runners implemented and tested on synthetic
fixtures only, including isolated NCN saved-checkpoint inference. PreScience
secondary subgroup analysis is integrated. A draft-plan assembler exists but
its complete real-artifact execution awaits prerequisites. No real global
final lock or held-out run exists.

## PreScience

`prescience_final_run.py` consumes the `prescience` entry in the existing global
three-dataset lock. It does not choose hyperparameters, fit models, select a
fixed-scorer variant, or resize the held-out sample from results.

Additional fields required beyond `final_evaluation_gate.py`:

- `runner_schema: 1`, `feature_names`: exact23-feature hybrid schema.
- `corpus_inputs`: workspace-relative `corpus` and `author_history` files,
  both declared as data artifacts. The corpus must be the separate official
  TEST companion corpus for the real final run.
- `query_sampling`: integer `count` (0 means census) and integer `seed`.
  Decide count from development resource measurements before final locking.
- `candidates`, `padding_seed`: explicit positive count and integer seed.
- `index_inputs`: workspace-relative `lexical` and `semantic` directories.
  Their manifests and every consumed vector/matrix/metadata file must be locked
  data artifacts. Their source SHA must equal the explicit corpus SHA.
- `inference`: explicit `repetitions` and `seed` for paired inference. Actual
  final policy:10000 repetitions, seed314159, as already specified in protocol.
- `output_directory`: one workspace-relative destination for the frozen run.
- `methods`: each contains `candidate_mode` (`full` or `no_content`) and a kind:
  - `learned`: `models` maps every model seed to its FrozenRanker spec. Paths
    from `frozen_ranker.model_spec` must be converted to workspace-relative paths.
    Include model and source development report in the appropriate artifact roles.
  - `feature`: one exact `feature` name for shared-pool scoring.
  - `standalone`: an existing named retrieval `ranking` from the engine.
  - `fixed`: a preselected `variant` from `prescience_fixed_scorer.VARIANTS`,
    evaluated on the declared candidate pool. Selection occurs on validation.
- `seeds`, `primary_metric`, `primary_comparisons`, `multiplicity` retain the
  global gate schema. Include the eight scientifically declared primary
  PreScience contrasts from SCIENTIFIC_REPORTS_PLAN, and all planned reporting
  methods/ablations. The runner is generic; it executes exactly the frozen
  family and never invents extra contrasts from final results.
- `subgroup_comparisons`: explicit secondary method pairs, required before
  test access. Uses the frozen inference repetitions/seed and writes
  `subgroups.json`: model-seed-averaged query differences, joint seed-author
  bootstrap, history strata, new/repeat Recall10 and direct effect differences.
  These are marginal secondary intervals; groups below20 authors are flagged.

All runtime source dependencies in the runner's `CODE` list must be included
under the `code` role. Freeze the actual environment and complete development
reports as well. This document is not a usable final configuration.

Example command shape, only after the real global lock is ready:

```powershell
& '.\work\venv\Scripts\python.exe' outputs/researchforge-study/scripts/prescience_final_run.py --lock <frozen-global-json> --run-id <frozen-run-id> --ledger <workspace-ledger-directory>
```

## Integrity and outputs

Authorization and full global-artifact verification precede corpus-label access.
Model/source bindings, feature schemas and index-corpus consistency are checked
before the corpus adapter reads labels. Every engine call uses `training=False`;
positive injection is prohibited. One writer lock protects the output.

Per-query feature files and predictions receive commit receipts; an interrupted
run resumes only with identical run ID/global lock/configuration and unchanged
committed artifacts. Completed replay verifies the full output inventory.
Final per-method/per-seed rankings are independently checked with pytrec_eval.
Query metrics are averaged across model seeds before author-cluster inference;
the frozen family receives Holm-adjusted tests. Fixed baselines repeated over
model seeds do not create independent observations.

Current integration tests cover real tiny-model inference in full/no-content
candidate modes, shared-pool features, standalone semantic and fixed scoring,
independent metrics, two-query seed averaging, interrupted resume, output
corruption, missing global lock and missing model binding. Temporary synthetic
locks are cleaned up; they do not authorize the real benchmark.

Remaining beyond this runner: prepare the actual final model/config manifest,
complete the distinct final semantic cache, execute all final benchmark runners,
produce final subgroup reports/plots, isolated costs and manuscript tables.

## OGB

`ogb_final_run.py` consumes the `ogbl-collab` entry after the same global gate.
It never calls the OGB dataset loader, fits a model, or adds validation edges.
Required additional fields:

- `runner_schema: 1`, `graph_cutoff_year: 2017`.
- `inputs`: exactly `edges`, `years`, `weights`, `nodes`, `split`, explicit
  workspace-relative files bound as data artifacts. For the real run, freeze
  the released training graph and the official2019 test split. All graph years
  must be at most2017; the released node matrix has128 columns.
- `negative_count`:100000 for the actual official split (small synthetic
  fixtures use100). `batch_size`: positive NCN inference batch size.
- `inference`: frozen integer `repetitions` (at least100) and `seed`.
- `output_directory`: dedicated workspace-relative final output folder.
- `methods`: `kind: tree`, with `track: graph` or
  `graph_plus_released_nodes`, and `models` mapping every seed to a locked
  LightGBM text file; `kind: ncn` with saved best checkpoints per seed; or
  `kind: feature` with an exact graph feature name from `ogb_validation.NAMES`
  or `released_node_cosine`. All model paths must have role `model`.
- Bind all files in `ogb_final_run.CODE` as code; NCN additionally requires
  `ncn_temporal.py` and pinned upstream `model.py`/`utils.py`. The actual
  environment artifact must document base and isolated package versions.

NCN needs a fresh process using the existing isolatedTorch2.5.1+cpu runtime.
The runner selects that runtime before importingTorch and rejects an already
loaded incompatible version. On this machine the sandbox may block read access
to the installed sparse extension; use the established reviewed read-access
execution path, without reinstalling packages or modifying results.

Per-method/seed predictions have atomic commit receipts and identity/SHA
checks. Interrupted runs reuse committed predictions; completed replay checks
all inventoried outputs. Hits@50 is independently calculated with strict `>`
against the50th-largest shared negative and checked against the OGB evaluator.
Source-cluster uncertainty uses seed-averaged per-edge success differences,
with Holm correction of the frozen OGB comparison family. This does not model
crossed graph dependence or uncertainty from the fixed negative pool.

Tests: `test_ogb_final_run.py` four synthetic integration tests cover graph and
node-feature trees, fixed scorers, gate-before-label loading, strict ties,
interrupted/completed replay and output corruption. `test_ogb_final_ncn.py`
adds a full isolated-runtime synthetic lock/run and compares saved batched NCN
predictions for three seed entries to direct unbatched checkpoint inference.
These tests establish software behavior only; they are not benchmark results.

Command shape after the real global freeze:

```powershell
& '.\work\venv\Scripts\python.exe' outputs/researchforge-study/scripts/ogb_final_run.py --lock <frozen-global-json> --run-id <frozen-run-id> --ledger <workspace-ledger-directory>
```

## exHarmony

`exharmony_final_run.py` consumes the global `exHarmony` configuration. Required:

- `runner_schema: 1`, exact `feature_names` from `exharmony_bm25_pilot.FEATURES`,
  positive `candidates`, `primary_judgment: authors`.
- `inputs`: exactly `queries`, `judgeable`, `split`, `authors`, `cite`, `simcite`.
  `split` must be bound with role `split`; the others as `data`.
- `index_directory`: filtered historical corpus index. Bind every entry in
  `INDEX_FILES`; stored BM25 manifest/stats hashes must agree.
- `query_sampling: {count, seed}`; zero means adapted-test census. All models
  share the selected queries. Components are built from ALL test query metadata
  so an unsampled bridge does not split a component.
- `methods`: learned saved LightGBM specs, exact feature list, model SHA and
  development-report provenance for every seed; or a fixed feature name.
- `inference: {repetitions, seed}`, explicit primary method pairs and output.

Candidate generation receives an EMPTY truth set and training=False. Missing
author profiles remain in raw relevance denominators; empty rankings are kept.
Authors, Cite and SimCite metrics are independently checked by pytrec_eval;
only Authors belongs to the primary inferential family. Model-seed-averaged
differences are clustered by query-author/title components. Shared cited-expert
dependence is not fully modeled. Per-query data/features and final outputs have
commit receipts/hash checks. Four synthetic tests cover these behaviors,
gate-before-label reads, committed resume, corrupt output and empty retrieval.

## Draft assembly and costs

`build_final_plan.py` requires completed expanded development, final semantic
audit, isolated pipeline costs and an explicit environment artifact. It assembles
saved model paths/schemas and selects the fixed scorer solely by expanded
seed42 validation nDCG1000. It requires an explicit PreScience query count and
development-based sampling rationale; it does not select these from final labels.
The output stage is `draft_pending_development_review`, which the global gate
REJECTS. Review actual evidence and all paths, then explicitly finalize via
`freeze_plan`. No user approval is newly required for this authorized local work.
Full builder integration remains to verify against completed real artifacts.

Assembler integration update27October: `test_build_final_plan.py` passes four
temporary synthetic fixture tests. All three configurations are assembled with
the actual AST feature/dependency schemas and saved-model hash checks; the
builder's draft remains unfreezable. A deliberately finalized temporary fixture
passes the global artifact inventory check. Tests reject a corrupted model,
missing prerequisite, failed lexical audit, incomplete development report and
missing/duplicate/nonfinite cost observations. This tests assembly, not real
model inference or benchmark performance. Actual real-artifact review remains
pending. Cost prerequisites now require both30-query/3-repeat pipelines with
all90 unique observations and three finite nonnegative model-scoring timings.

`capture_environment.py` records base package versions and the separate NCN
overlay manifest in `environment-current.json`; recapture before freeze if
packages change. `run_final_text_when_ready.py` waits for the completed TRAIN
semantic audit, builds label-free final embeddings and audits hash/finite/norms.
Expanded development resumes after this final semantic audit. The cost queue
waits for that audit, availableRAM and no competing study workers. It then runs
`pipeline_cost.py` on30 common development queries with3 repeats, saved-feature
parity, full/no-content retrieval+features, three saved model scorers and20ms
RSS sampling. It does not infer preprocessing/training compute from paused logs.

## Serial final execution

`run_frozen_study.py --lock <reviewed-global-lock> --run-id researchforge-final-v1
--wait-for-lock` waits for a separately reviewed freeze, then executes OGB,
exHarmony and PreScience sequentially, verifies completed report bindings, and
exports verified tables and figures. It never assembles or approves its own
configuration. Resource admission and process-tree monitoring protect the
local RAM/disk reserves. State and logs are in `work/frozen-study-*`; see STATUS
for active PIDs before starting anything. Partial exports and changed inputs
stop execution for inspection. Completed stages resume only with matching lock
and run identity. Four scheduling tests pass; actual held-out execution remains
pending. Completion of this queue excludes manuscript and visual review.

`preflight_saved_models.py` checks actual completed development models without
opening final labels. The27October audit loaded18LightGBM models, checked ordered
schema widths/objectives and development-selected trees/leaves, and predicted
three synthetic rows. It also checked3NCN development receipts/source hashes and
inventoried checkpoints without deserialization. The resulting
`data-audit/final-model-preflight.json` is interface/integrity evidence, not a
performance evaluation. Expanded PreScience models require their own checks
when available; the global freeze still binds all actual files independently.
