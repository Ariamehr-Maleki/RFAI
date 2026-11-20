# ResearchForge ranking study — reviewer repository

**Ranking gains in scientific collaborator forecasting depend on evidence and comparator choice**  
Ariamehr Maleki · Amirhossein Keyhanipour  
Intelligent Information Fusion Lab, Faculty of Engineering, College of Farabi, University of Tehran, Tehran, Iran

This is the compact, auditable source release for an offline ranking study motivated
by ResearchForge. It is **not** a validation of the original eight-agent application.
The original application is at [tuba89/ResearchForge-AI](https://github.com/tuba89/ResearchForge-AI).

## Start here    
    
Python 3.10+ is sufficient for these read-only checks; no installation, network,
API key, corpus download or model fitting is required:

```sh
python review.py audit
python review.py results
python -m unittest discover -s tests -v
```

The audit verifies repository payload hashes, preserved experiment sources and
**568 full-precision numerical rows** against final JSON reports. This is an
aggregate-evidence check, not a rerun of all saved predictions.

## What the evidence says

| Task / metric | Practical ranker | Matched classifier | Strong comparator |
|---|---:|---:|---:|
| PreScience / nDCG@1000 | 0.488 | 0.478 | Official harmonized scorer **0.507** |
| ogbl-collab / Hits@50 | 0.524 | 0.526 | NCN **0.536** |
| Adapted exHarmony / Authors nDCG@10 | 0.121 | 0.117 | BM25 **0.126** |

These metrics are not comparable across rows. Boldface identifies the listed
comparator, not a common significance rule. BM25's higher point estimate does not
establish a significant advantage over the ranker. All confidence intervals,
correction families and limitations remain in the reports. The official
PreScience comparison is a **post-result supplementary analysis**, not an original
untouched confirmation. Negative results are retained.

## Additional cohort and independent audit — 20 November 2026

An additional **1,000 disjoint query identities** were evaluated with the frozen
models, without fitting or tuning. This is a post-result within-release check,
not an external replication; 16 seed authors overlap the original cohort.

| Additional PreScience sample | nDCG@1000 | New-collaborator Recall@10 |
|---|---:|---:|
| LambdaMART | 0.479 | 0.026 |
| Matched classifier | 0.469 | 0.029 |
| Official harmonized scorer | 0.500 | 0.034 |

The paired ranker-minus-classifier difference is +0.010 (95% CI +0.005 to +0.015),
while ranker-minus-official is −0.021 (−0.029 to −0.013). The two comparisons use
their own Holm family; new-collaborator recall is descriptive. Candidate policies
are method-specific for the official comparison. These results retain the
original positive and negative findings.

Independent checks reproduced 213 original prediction/score files, 15,534 saved
PreScience/exHarmony model rankings and 16 original inferential contrasts. For the
additional cohort, all 6,000 learned rankings and saved metrics passed independent
replay; 21 raw feature matrices were reconstructed exactly. This does not mean
every raw feature, every baseline model or every subgroup analysis was rerun.

See [additional-cohort reproduction](docs/ADDITIONAL_COHORT.md),
[full-precision results](results/additional-cohort/results.json) and
[per-display coverage](docs/JOINT_COVERAGE.md). Full caches and predictions remain
in local archives, not this Git repository. No public archive DOI is asserted.

## Repository map

| Path | Purpose |
|---|---|
| `src/rfai_review/` | Small, documented reviewer CLI; no experiment side effects |
| `tests/` | Failure-path checks and evidence consistency safeguards |
| `results/` | Exact aggregate reports, contrasts, cost and subgroup evidence |
| `experiments/outputs/researchforge-study/scripts/` | Byte-preserved original experiment modules and fixtures |
| `experiments/outputs/researchforge-study/review-20261107/` | Official-scorer extension and its pinned upstream source |
| `experiments/outputs/researchforge-study/review-20261120/` | Independent audits, frozen additional-cohort protocol and exact execution sources |
| `results/additional-cohort/` | Full-precision new sample summary and effect-plot values |
| `provenance/review-20261120/` | Independent verification and per-display coverage receipts |
| `provenance/` | Original lock, archive inventory, source identity and release scope |
| `docs/` | Reproduction steps, method map, data requirements and limitations |

Read [reproduction instructions](docs/REPRODUCE.md), the
[module map](docs/MODULE_MAP.md), and [data and limitations](docs/DATA_AND_LIMITATIONS.md).
The hierarchy separates the clean reviewer entry point from preserved scientific
execution code. Reformatting those frozen files would invalidate their recorded
hashes; their locations and bytes are intentionally retained.

## Scope and availability

This repository includes code and compact results. Raw provider datasets,
large prediction files, fitted-model archives and Python environments are not
Git payloads. The existing full local archives are listed in provenance and
are intended for a later archival deposit. **No Zenodo DOI exists for this release.**
Consequently, a fresh clone supports the quick audit and fixture tests, while a
complete exact prediction replay also requires the omitted archives and provider
assets. See the explicit reproduction tiers; do not interpret a passing quick audit
as full replay or as publication acceptance.

## Research scope

- Standard LightGBM LambdaMART; no new optimization algorithm.
- Historical publication text, network evidence and recency; three frozen model seeds.
- Shared-pool factorial and separate candidate-changing content ablation.
- Separate held-out tasks; paired uncertainty and declared Holm families.
- No human evaluation, prospective collaboration outcomes or eight-agent benchmark.
- No claims of superiority over unexecuted CoBERT, TEDRec or other recent models.

OpenAI Codex assisted implementation, execution, analysis and manuscript preparation.
The human authors remain responsible for review and approval. Licensing status and
third-party rights are stated in [LICENSE.md](LICENSE.md); no blanket relicensing
of archived or upstream code is implied.
