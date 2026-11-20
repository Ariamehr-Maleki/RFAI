# Additional-cohort and independent-review reproduction

The original study is unchanged. The additional sample was selected after the
original results were known, with frozen query identities, models and two planned
contrasts before its inference. The same provider release/history is reused.
This repository preserves exact scientific scripts and compact receipts, not the
complete 1,000-query cache and prediction collection.

## Quick review

Run `python review.py audit`, `python review.py results` and the wrapper tests
from the repository root. Results include the original three tasks and the
additional cohort. The compact audit checks hashes and aggregate consistency;
it does not execute the independent scientific replay.

## Full evidence layout

Obtain the three archives listed below through actual author-provided access.
They have been prepared locally, but no public deposit is available yet:

1. `researchforge-reproducibility.zip`
2. `ResearchForge_Review_Extension_20261108.zip`
3. `ResearchForge_Additional_Cohort_20261120.zip`

Use a new isolated directory. Preserve the paths stored inside the archives;
do not overwrite the evidence copies when rerunning audits. The review scripts
expect the first two ZIP files under `outputs/researchforge-study/deliverables/`
and the third archive's extracted `outputs/researchforge-study/review-20261120/`.
The original archive provides saved models and original caches; the third provides
the additional cache, predictions and derived OGB cluster identifiers.

From the isolated root, use the recorded scientific environment to run, in order:

```sh
python outputs/researchforge-study/review-20261120/audit_archive_evidence.py
python outputs/researchforge-study/review-20261120/replay_cached_models.py
python outputs/researchforge-study/review-20261120/check_inference_families.py
python outputs/researchforge-study/review-20261120/audit_fresh_rankers.py
python outputs/researchforge-study/review-20261120/audit_fresh_results.py
```

Raw-feature reconstruction with `audit_fresh_overlap_features.py` additionally
requires pinned provider inputs, indices and encoder assets. Raw inference and
training also require these omitted assets. The historical launch/supervisor
scripts are preserved as evidence, not recommended as a portable task scheduler;
they contain Windows operational checks. Do not launch multiple writers.

## Verified scope

The original metric audit covers 213 prediction/score files. Exact cached-model
ordering checks cover 6,000 PreScience rankings and 9,534 exHarmony rankings at
their retained depths. Sixteen paired inferential contrasts were independently
recomputed. Quarter/subgroup resampling and all raw-data training were not newly
repeated. Additional-cohort checks cover 6,000 learned rankings, all official
saved-output metrics, both contrasts and 21 reconstructed raw feature matrices.
The official algorithm was executed, but not independently reimplemented.

The Windows write/overlap incident is retained in `OVERLAP_INCIDENT.md`; exact
source versions and full overlap-window feature parity are preserved. These
operational failures were not erased or treated as extra statistical trials.

See SOURCE_LICENSES.md before redistribution. Both-author manuscript approval,
large-archive access and an explicit general software reuse license remain
separate from this authorized compact code update.
