# Data, methods and limits

| Task | Inputs and labels | Split and evaluation |
|---|---|---|
| PreScience | Historical papers, coauthors, references, citation snapshots; other eligible target-paper authors as positives | Temporal development/final periods; 1,000 sampled final queries, 989 seed authors; nDCG@1000 |
| ogbl-collab | Scholarly coauthorship graph; a separate released-node-vector track for NCN/vector models | Official train/validation/test; 46,329 test positives, 100,000 shared negatives; Hits@50 |
| Adapted exHarmony | Publication retrieval with authorship and citation-derived expert relevance | Whole author/title components assigned to splits; 1,589 held-out queries; Authors nDCG@10 primary |

Providers: [PreScience](https://huggingface.co/datasets/allenai/prescience),
[Open Graph Benchmark](https://ogb.stanford.edu/docs/linkprop/),
[exHarmony](https://huggingface.co/datasets/Reviewerly/exHarmony).
Follow the current provider licenses and the exact release/checksums in the locks.
Raw data are not redistributed in this repository.

The practical PreScience pipeline uses 23 features: 3 history controls, 7 content,
9 network, 2 explicit-time and 2 composite scores. The shared-pool factorial omits
the composites, and its largest cell has 21 features. The separate no-content
pipeline has 16 features and changes retrieval too. These are different interventions.
Models are standard LightGBM rankers/classifiers; no novel optimizer is claimed.

## Limits that must accompany the results

- Coauthorship is a retrospective proxy, not a judgment of compatibility, consent,
  availability, invitation acceptance or collaboration success.
- New-partner recall stays low; aggregate gains can favor repeated partners.
- The official harmonized PreScience scorer beats the ranker. The earlier
  description-based reconstruction is explicitly an approximation.
- The official-source comparison is supplementary and post-result. Temporal
  masking does not remove its advantage; raw-input scores are provenance sensitivities.
- The OGB objective and sampled training groups differ from its global-negative
  evaluation. NCN has different inputs and a larger checkpoint-selection budget.
- exHarmony is an adapted undated expertise task, not a temporal collaborator
  benchmark or a reproduction of the provider's exact published split.
- Marginal cluster-bootstrap intervals are not simultaneous intervals. Model-seed
  SD is not a confidence interval; some crossed network dependence remains.
- Warm cost excludes preprocessing, index construction, encoding and training.
- The original eight agents, generated explanations/proposals and outreach are
  outside the evaluated path. No human evaluation was conducted.
