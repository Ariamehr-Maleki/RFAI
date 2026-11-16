# Reproducing the November scientific-review extension

This is an additive extension to researchforge-final-v1, not a replacement of
the original frozen experiment. It uses the same 1,000 PreScience queries and
three frozen practical rankers. It fits no models. Results were designed and
computed after the original final results were known.

## Evidence to retain

Retain the original `researchforge-reproducibility.zip` and its receipt, together
with the dated review extension archive and its receipt. The extension preserves
workspace-relative paths under `outputs/researchforge-study/`. Extract both into
the same workspace layout. The small professor-review PDFs are presentation
artifacts; they are not substitutes for the executable evidence.

The original archive SHA256 is
`94dc644363e1a62ae619664af5359c56135ab76e4b9156a389cef2180845dedb`.
The original global lock SHA256 is
`0f2240d82c5013a39099cadd15b67bb2a1a1298e96fce33888dde463fd5ad091`.
The supplementary lock SHA256 is
`e8ac71a57177a7cef04e82bddeb354886d5b44826484fa4dacb9db6e0185ea39`.

Raw provider data and Python environments are not redistributed. Retrieve data
from their original providers using the original study's source instructions,
then verify the raw-input hashes against the supplementary lock. Its file paths
record this execution environment; moving input files requires an explicitly
documented path adaptation, not silently changing scientific settings.

## Source and input policies

The official source is pinned at PreScience commit
`e54f6a5441aa389c18acfc11080facea6a95c1d5`. The scorer Git blob is
`1dac60a8a1f88dc55fc6d5a57111cd255b04a579`. Exact retrieved strings, local snapshot
hashes, AST comparisons and the upstream license are included. Snapshot files
have an additional terminal newline; semantic parity is checked separately from
byte identity. Scoring constants and function bodies are taken directly from
the pinned source, rather than reconstructed from prose.

`REVIEW_PROTOCOL.md` defines published-input, historical-input and harmonized
historical-input tracks. All three must be reported, with no selection of the
best result. The published-input track is a provenance sensitivity because
released citation trajectories can contain later information. The historical
adapter relies on the pinned data builder's 30-day cumulative-bin convention;
it is not an independent citation-event reconstruction.

## Commands from the workspace root

Use the experiment interpreter/version recorded in `supplementary-lock.json`.
The exact run used `work/venv/Scripts/python.exe` on Windows. It requires NumPy,
PyArrow, psutil and pytrec_eval, plus the original `scripts/final_statistics.py`.
Set `PYTHONHASHSEED` before starting Python because upstream padding uses hash
and set iteration. Package versions are retained with the original runtime.

```powershell
$env:PYTHONHASHSEED = '0'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
work/venv/Scripts/python.exe -m unittest discover -s outputs/researchforge-study/review-20261107 -p test_official_comparator.py -v
work/venv/Scripts/python.exe -m unittest discover -s outputs/researchforge-study/review-20261107 -p test_indexed_helper.py -v
work/venv/Scripts/python.exe -m unittest discover -s outputs/researchforge-study/review-20261107 -p test_bounded_helper.py -v
work/venv/Scripts/python.exe -u outputs/researchforge-study/review-20261107/run_bounded_with_memory_wait.py
```

The indexed helper preserves the returned history IDs and order. It checks the
first 1,000 calls and every 1,000th subsequent call against the upstream helper;
it does not check every distinct call. Its hash and sampling rule are separately
bound in `runtime-addendum.json` and `runtime-bounded-addendum.json`. The initial
indexed cache was unbounded and subsequent resource-reserve exits prompted a
value-preserving bounded LRU cache: at most 10,000 author/history entries and
1,000,000 publication slots. Five additional indexed/bounded tests cover order,
boundary and eviction behavior; the original seven temporal/metric tests are
unchanged. Earlier parity receipts and failure logs remain preserved. The outer
launcher waits in 15-second intervals below the existing 450 MiB RAM reserve and
requires 1,800 MiB before loading. It retains loaded data during a resource wait.
This operational change is bound in runtime-memory-wait-addendum.json; wait totals
are recorded separately and are not experimental latency evidence. The scorer
itself is unchanged. Resource-reserve
interruptions retain completed query/track checkpoints. Resume using the same
command; a mismatching input lock is an error. Do not launch concurrent copies.
Operational run times include adapters and desktop contention; they are not a
fair latency comparison with the original isolated warm-cost experiment.

To rebuild summary statistics from all 3,000 existing query/track checkpoints:

```powershell
$env:PYTHONHASHSEED = '0'
work/venv/Scripts/python.exe -u outputs/researchforge-study/review-20261107/official_comparator.py --summarize-only
```

Use the unwrapped command for summary-only work so that it does not overwrite
the completed runtime-helper parity receipt. Each supplementary prediction file
has 1,000 records. Independent pytrec_eval verifies five metrics per record.
Paired author-cluster uncertainty uses the frozen ranker's three-seed mean and
10,000 draws. Holm correction covers exactly three new nDCG contrasts and is
separate from the original primary families.

The separate dependence diagnostic is reproduced with:

```powershell
work/venv/Scripts/python.exe -u outputs/researchforge-study/review-20261107/robustness_review.py
```

It joins queries sharing any observed target-team author. This addresses more
overlap than source-author clustering but does not remove every dependency
arising from shared candidates or historical networks. Quarter analyses are
descriptive and post-result.

## Document build and boundaries

`build_review_revision.py` requires complete independently checked comparator
outputs and source/runtime parity receipts before producing revised text.
`render_review_documents.py` uses the bundled document runtime with ReportLab,
Pillow and pypdf. Document formatting does not alter scientific evidence.

No public deposit, repository push or journal submission is part of this local
extension. Human authors must approve scientific claims, authorship, affiliations,
funding, conflicts and AI-use disclosure, and arrange reviewer access to code and
data before submission. The AI-assisted review is not independent peer review.
