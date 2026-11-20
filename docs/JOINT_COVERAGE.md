# Joint archive coverage and reviewer-access audit

Both original archives and the new additional-cohort archive are complementary. Every payload hash passed a new streaming read-back. No files were uploaded and no DOI was assigned.

| Display | Evidence | Metric/model replay | Access boundary |
|---|---|---|---|
| Table 1 | references.json | Interpretive literature synthesis, not a metric result; references.json supplies metadata but does not independently validate every synthesized claim. | Full archives remain local |
| Figure 1 | figure0_architecture.pdf, fig_architecture.py | Not an empirical metric display | Full archives remain local |
| Table 2 | prescience-report.json, official-comparator-results.json | PASS: independent metrics from archived rankings/scores; see independent-metrics.json | Full archives remain local |
| Figure 2 | prescience-report.json, official-comparator-results.json, primary_contrasts.csv | PASS: all relevant declared contrasts independently replayed, including marginal intervals and within-family Holm; see independent-inference.json | Full archives remain local |
| Figure 3 | prescience-report.json | PASS: independent metrics from archived rankings/scores; see independent-metrics.json | Full archives remain local |
| Table 3 | ogb-report.json, exharmony-report.json | PASS: independent metrics from archived rankings/scores; see independent-metrics.json | Full archives remain local |
| Table 4 | prescience-report.json, ogb-report.json, exharmony-report.json, pipeline-cost.json | Interpretive application recommendations based on checked performance and archived timing evidence; no deployment-benefit experiment. Timing quantiles were not remeasured. | Full archives remain local |
| Table S1 | prescience-report.json | Not an empirical metric display | Full archives remain local |
| Table S2 | prescience-report.json | PASS: independent metrics from archived rankings/scores; see independent-metrics.json | Full archives remain local |
| Table S3 | prescience-report.json | PASS: independent metrics from archived rankings/scores; see independent-metrics.json | Full archives remain local |
| Table S4 | exharmony-report.json | PASS: independent metrics from archived rankings/scores; see independent-metrics.json | Full archives remain local |
| Table S5 | primary_contrasts.csv, prescience-report.json, ogb-report.json, exharmony-report.json | PASS: all relevant declared contrasts independently replayed, including marginal intervals and within-family Holm; see independent-inference.json | Full archives remain local |
| Table S6 | official-comparator-results.json | PASS: all relevant declared contrasts independently replayed, including marginal intervals and within-family Holm; see independent-inference.json | Full archives remain local |
| Table S7 | dependence-robustness.json | Source rankings/metrics verified; this quarter/subgroup resampling was not independently rerun in the November 20 audit | Full archives remain local |
| Figure S1 | subgroups.json | Source rankings/metrics verified; this quarter/subgroup resampling was not independently rerun in the November 20 audit | Full archives remain local |
| Figure S2 | pipeline-cost.json | Saved timing observations available; original hardware workload cannot be recreated from predictions | Full archives remain local |
| Table S8 | ogb-report.json | PASS: independent metrics from archived rankings/scores; see independent-metrics.json | Full archives remain local |
| Table S9 | main.tex, supplement.tex | Not an empirical metric display | Full archives remain local |
| Table S10 | fresh-results.json, fresh-independent-results-audit.json | PASS: all additional metrics and two contrasts independently recalculated; all 6000 model rankings replayed; source fresh-independent-results-audit.json | Full archives remain local |
| Figure S3 | prescience-report.json, official-comparator-results.json, fresh-results.json, figureS3_additional_cohort.csv | PASS: all additional metrics and two contrasts independently recalculated; all 6000 model rankings replayed; source fresh-independent-results-audit.json | Full archives remain local |

The JSON manifest supplies exact file paths, SHA256 values, archive matches, execution level, provider dependencies and access/licensing status. A source-table hash does not by itself establish statistical correctness.

New audit implementation corrections are retained: the initial filename parser split `.sum` as though it were a seed suffix, and the first exHarmony check mistakenly treated top-100 rankings as full candidate pools. These audit failures did not change the original experiments. The corrected audit joins the 1,589 shared query caches by qid and recalculates candidate recall from all 1,000 candidates. Cached-model replay compares the complete retained output depth (1,000 for PreScience, 100 for exHarmony), with descending score and ascending identifier tie-breaking on the stored float32 feature matrix. Raw model-score parity is not asserted because prediction files retain rankings rather than per-author scores.
