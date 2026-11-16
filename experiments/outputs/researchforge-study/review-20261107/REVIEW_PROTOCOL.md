# Post-result scientific review and supplementary evaluation

Date: 7 November 2026. Authorized by the user to perform a referee-style review, repair findings, and resolve the official PreScience comparator.

The 29 October locked evaluation and archived deliverables remain immutable. This review occurs after those test results were observed. All new results are supplementary post-result analyses, never a newly untouched confirmatory test. No training or selection on final labels is permitted. No paid services, messages, repository push or journal submission.

## Review questions
1. Are task definitions, information access, baselines, metrics and uncertainty defensible?
2. Does the manuscript distinguish retrospective forecasting from successful real-world collaboration, and primary collaborator evidence from complementary tasks?
3. Is the claimed contribution specific and supported, including negative results?
4. Can the official v2 fixed scorer be executed rather than approximated?

## Newly discovered official source
The main branch still points to 47384ed79c48cfef610f30af12ba883f2a0093cb. Branch enumeration found baseline_autoresearch.py on agent, iclr-refresh and rebuttal-F-bio-physics with identical Git blob 1dac60a8a1f88dc55fc6d5a57111cd255b04a579. Pin iclr-refresh commit e54f6a5441aa389c18acfc11080facea6a95c1d5. Store its source, dataset helper, citation construction code and license.

Static inspection reveals previously unavailable team-size weights, different expansion windows, citation weighting, in-place score propagation and a different activity multiplier. It also reveals untruncated citation_trajectory summation and reference dereferencing without a reference-date check. These require a temporal audit before treating literal source execution as a leakage-controlled comparator. This is a source-level risk, not yet a measured impact claim.

## Supplementary comparison fixed before new predictions
Use all the original 1,000 PreScience query identities and the original frozen practical ranker predictions. Do not replace or retune any frozen model. Run the pinned scorer function itself, with k=1,000, on:
- published-input execution: raw released author/history order and citation trajectories, preserving upstream function behavior;
- historical-input execution: same raw authors/history, but hide same/future papers and truncate 30-day citation-trajectory bins to endpoints no later than the query cutoff;
- harmonized historical-input execution: the preceding temporal mask plus study author deduplication and date/ID history ordering.

The latter two are explicitly adapted inputs to the official scorer, not byte-for-byte published benchmark reproduction. All tracks retain fixed upstream constants; no best-track selection. Freeze Python hash seed to 0 and record Python version because upstream padding uses hash(seed) and set iteration. No target author labels enter candidate construction/scoring; only the preselected source author and date do. Save raw predictions, audit accesses, metrics and hashes. The raw track is a provenance sensitivity, not primary evidence of historical deployability.

Compare original frozen ranker seed-mean versus each track using the same saved labels, nDCG@1000 plus shallow/new/repeat recall and candidate recall. Report paired seed-author cluster intervals with 10,000 resamples and Holm adjustment for this separate three-contrast supplementary family. Independently recompute ranking metrics. Do not merge these p-values into the original primary family. An additional endpoint for shorter-prefix ranking may only be descriptive.

Meaningful tests: temporal citation-bin boundaries, future-reference masking, exact upstream helper/function extraction, raw-versus-adapted equality where no input transformation is necessary, and candidate/label separation. Inspect resources; one single-thread worker, checkpoints and a disk reserve. Update status after runs.

## Deliverables
Referee report and action matrix; upstream provenance and implementation comparison; completed supplementary comparator results (or precise failure); revised manuscript and supplement; updated professor report/PDFs in a distinct dated folder. Preserve all negative outcomes and disclose remaining limitations. Automated self-review does not substitute for named authors' approval.
