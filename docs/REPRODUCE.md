# Reproduction tiers

Run commands from the repository root. Do not mix development results with held-out results.

## Tier 1 — compact evidence audit

```sh
python review.py audit
python review.py results
python -m unittest discover -s tests -v
```

Only the Python standard library is needed. The root MANIFEST checks all distributed
payloads. The original source inventory checks byte identity against the original
archive independently. Numerical exports are checked against the three final reports.
This verifies consistency, not the correctness of unavailable raw inputs.

## Tier 2 — synthetic scientific fixtures

The experimental interpreter was Python 3.12.14. Exact versions are recorded in
`requirements-experiment.lock.txt`; full environment snapshots are in provenance.
Use an isolated environment. On a platform where a pinned wheel is unavailable,
record any adaptation rather than silently calling it the original environment.

```sh
python -m venv .venv
# Activate .venv using the command appropriate to your shell.
python -m pip install -r requirements-experiment.lock.txt
python review.py fixtures
```

Fixtures exercise temporal eligibility, BM25 behavior, test-access gates,
seed averaging, multiple-comparison correction, frozen-model round trips, and
the official-scorer adapters. They use synthetic inputs and do not fetch real
datasets or authorize new test tuning. The code was checked locally with the
retained experimental environment; CI runs the dependency-free Tier 1 checks.

## Tier 3 — complete exact replay (additional assets required)

The original local archive is `researchforge-reproducibility.zip`, SHA256
`94dc644363e1a62ae619664af5359c56135ab76e4b9156a389cef2180845dedb`.
It contains model/prediction assets excluded from this compact Git release.
The later review-extension archive contains the official-scorer predictions.
Neither archive is represented as publicly deposited yet. Obtain those assets
from the authors / future deposit before claiming exact replay availability.

1. Extract original artifacts into `experiments/`, preserving paths beginning
   `outputs/researchforge-study/`. Verify the original manifest and receipt.
2. Obtain provider corpora and checkpoint assets using the source URLs recorded
   by `download_data.py` and the locks. Compare SHA256 values; stop on mismatch.
3. Rebuild or restore the label-free lexical, semantic and graph indices following
   the module map. Do not use final labels for candidate or model selection.
4. Restore the pinned NCN source and its isolated CPU runtime if replaying the
   neural baseline. NCN has a different input/budget track from the graph-only trees.
5. Work from `experiments/`. The preserved `run_frozen_study.py` checks the global
   lock and original output run ID before reusing final artifacts. Use the run ID
   recorded in the JSON reports; do not overwrite them with a new run.
6. Follow `review-20261107/REPRODUCTION.md` for the supplementary official-source
   audit. Set `PYTHONHASHSEED=0` before Python starts and preserve all three tracks.

Example existing-artifact validation command, after all missing assets are restored:

```sh
cd experiments
python outputs/researchforge-study/scripts/run_frozen_study.py --lock outputs/researchforge-study/final/global-lock.json --run-id ORIGINAL_RUN_ID_FROM_REPORT
```

The explicit run-ID token must be replaced with the saved report value. A fresh
training experiment is a new study execution, not a byte-identical replay of the
old fitted models. Preserve its own configuration lock and outputs.

## Resource and portability boundaries

The original experiments ran on Windows with approximately 12.77 GB RAM and four
logical CPUs. Run one heavy encoding/training job at a time. Disk requirements
depend on restored corpora and caches; the compact repository size is not an
estimate of full replay storage. Absolute paths in provenance describe the actual
run. Preserve relative layouts and document any path-only adapters. A hardware
change does not permit interpreting warm latency as a portable service guarantee.

## Chronology

The original test configuration was locally frozen before final scoring. The
November official-source and additional dependence checks were specified after
the original outcomes were known. The November 16 manuscript revision introduces
no new fitting, model selection, test cohort changes or altered numerical results.
The local freeze is not a public prospective preregistration.
