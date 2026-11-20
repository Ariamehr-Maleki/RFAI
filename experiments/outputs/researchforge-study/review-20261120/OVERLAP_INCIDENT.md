# Operational overlap, 20 November 2026

The V2 worker stopped after 84 committed queries when Windows denied replacement
of its progress file. Two chats then independently prepared an operational fix.
The article chat launched the unique-temp-file retry wrapper while the reviewer
chat temporarily edited the underlying runner and launched another supervisor.
The reviewer stopped its duplicate processes (20860, 22372, 6560). The remaining
article supervisor was 9444 (launcher 20208), with worker 17932 (launcher 14488).
The shared supervisor-status file briefly described the stopped process; process
inspection, rather than that stale status file, established the live execution.

The active wrapper imported temporary runner SHA256
`437691d2f50b7705149f1afed0ff1ec55777787bdc38eb56474519fb2699fd77`.
The reviewer subsequently restored the on-disk V2 runner, SHA256
`a5df51a12c588c8f34867ad4eb7a870d0f06d1a6a18cd20675287f81bf6a2923`.
The temporary source was reconstructed from the recorded patch, independently
hashed, and preserved as `fresh_cohort_v3_imported_archive.py`. Its exact hash
matches both V3 locks. `v2-v3-operational.diff` shows changes confined to the
JSON writer, its uuid import, and operational prelock provenance; sampling,
feature construction, models, ranking, metrics and statistical settings match.
The wrapper overrides the JSON writer after import.

The other V3 supervisor is abandoned, but the temporary source and its lock
remain necessary evidence of the code actually imported by the active worker.
Thus the V3 run must not be described as executing byte-identical V2 source.
The two workers may both have written queries 85–102. Individual writer identity
is not established. No observed output or result is selected on performance.

At 11:58 UTC, 427 committed JSON/receipt/NPZ triples had matching checksums and
sample identity, with zero failures. This verifies internal file consistency,
not scientific equivalence by itself. The immutable snapshot is retained.

Before interpretation, the follow-up V4 operation must verify all committed
triples, independently replay all six saved model rankings for every query,
independently recompute ranking metrics, and reconstruct features for every query
in the suspected overlap window. V4 starts only after the V3 supervisor and its
worker exit, uses the original V2 source and unchanged sample, and has an exclusive
process guard. Existing locks and failed-attempt logs are retained unchanged.
The old wrapper is expected to reject its next stage because the source hash
changed; that rejection is a provenance safeguard, not a new scientific failure.
