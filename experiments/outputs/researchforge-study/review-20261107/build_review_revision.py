"""Build revised author-review sources from immutable evidence, after completion."""
from pathlib import Path
import json
import re
import hashlib
import shutil

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
DEST=STUDY/'deliverables/review-20261108'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def effect(r):return f"{r['delta']:+.6f} (95% CI {r['ci95'][0]:+.6f} to {r['ci95'][1]:+.6f})"
def replace_once(s,old,new):
    if s.count(old)!=1:raise ValueError('Expected exactly one replacement: '+old[:80])
    return s.replace(old,new,1)

def main():
    r=read(HERE/'official-comparator-results.json')
    if not r['status'].startswith('Complete') or r['independent_metric_max_error']>1e-12:raise ValueError('Completed verified comparator required')
    audit=read(HERE/'final-output-audit.json')
    if audit['status']!='passed' or audit['report_sha256']!=sha(HERE/'official-comparator-results.json'):raise ValueError('Completed cross-cohort audit required')
    dep=read(HERE/'dependence-robustness.json');parity=read(HERE/'source-parity.json')
    if not all(v['same_ast'] for v in parity.values()):raise ValueError('Upstream AST mismatch')
    runtime=read(HERE/'runtime-bounded-parity.json')
    if runtime['helper_calls']<1000 or runtime['checked_calls']<1000:raise ValueError('Completed runtime parity evidence required')
    if runtime['runtime_addendum_sha256']!=sha(HERE/'runtime-bounded-addendum.json'):raise ValueError('Runtime addendum mismatch')
    DEST.mkdir(parents=True,exist_ok=True)
    labels={'published_input':'Official scorer, published inputs',
            'historical_input':'Official scorer, historical inputs',
            'harmonized_historical_input':'Official scorer, harmonized historical inputs'}
    full=r['frozen_ranker_metrics'];safe=r['results']['harmonized_historical_input']['metrics']
    table='| Method | nDCG@1000 | Recall@10 | New Recall@10 | Candidate recall |\n|---|---:|---:|---:|---:|\n'
    table+=f"| Frozen practical LambdaMART | {full['ndcg1000']:.6f} | {full['recall10']:.6f} | {full['new_recall10']:.6f} | {full['candidate_recall']:.6f} |\n"
    for track,name in labels.items():
        m=r['results'][track]['metrics']
        table+=f"| {name} | {m['ndcg1000']:.6f} | {m['recall10']:.6f} | {m['new_recall10']:.6f} | {m['candidate_recall']:.6f} |\n"
    c=r['contrasts']['harmonized_historical_input']
    corrected=c['holm_p_three_supplementary_contrasts']<0.05
    if c['ci95'][0]>0 and corrected:verdict='The supplementary harmonized comparison favors the frozen ranker under the examined input policy and declared three-contrast correction.'
    elif c['ci95'][1]<0 and corrected:verdict='The supplementary harmonized comparison favors the official fixed scorer after the declared three-contrast correction; the ranker cannot be presented as the strongest evaluated collaborator pipeline.'
    else:verdict='The supplementary harmonized comparison does not establish an advantage of either method under the declared three-contrast correction; this is not an equivalence finding.'
    persists=all(r['contrasts'][t]['ci95'][1]<0 and r['contrasts'][t]['holm_p_three_supplementary_contrasts']<.05
        for t in ['historical_input','harmonized_historical_input'])
    persistence=('The official scorer retains its advantage in both historical-input tracks. The temporal audit does not explain away that advantage.' if persists else '')
    extension=f'''### 5.3 Supplementary audit of the official fixed scorer

After the original final results had been observed, a branch-level review found
the official Autoresearch scorer on non-default branches of the PreScience
repository. We pinned commit e54f6a5441aa389c18acfc11080facea6a95c1d5 [12].
The executable source contains operations absent from the earlier prose
reconstruction: team-size weighting, longer expansion histories, citation
weighting, in-place propagation and a different activity multiplier. Therefore,
the original approximation's score must not stand in for this implementation.

We fixed three additional input tracks before obtaining their predictions,
retaining the original 1,000 queries, frozen ranker models and k=1,000. The
published-input track executes the pinned scorer with raw released histories,
author lists and citation trajectories. The historical-input track hides papers
dated on or after the query and truncates cumulative citation bins to endpoints
no later than the query. A third track additionally applies the study's author
deduplication and date/ID history ordering. These are input adaptations to the
official scoring function, not a claim of exact reproduction of its published table.

The input audit matters because the scorer sums full citation trajectories for
second-hop papers and dereferences cited papers without a date check. The pinned
data-construction code defines trajectory bins as cumulative counts at publication
plus 30-day offsets. Using the whole released trajectory can expose later
information. The historical adapter retains the upstream sum of the available
prefix; it does not replace it with a newly chosen citation statistic. This
source-level finding does not establish what additional preprocessing was used
for the original paper's reported results.

{table}

{verdict} The frozen ranker minus harmonized historical scorer difference is
{effect(c)}, with Holm-adjusted p={c['holm_p_three_supplementary_contrasts']:.6f}
within this separate three-comparison supplementary family. The corresponding
published-input and historical-input differences are
{effect(r['contrasts']['published_input'])} and
{effect(r['contrasts']['historical_input'])}, respectively.

{persistence}

In published-input execution, {r['results']['published_input']['queries_accessing_unavailable_citation_bins']}
of 1,000 queries read a citation trajectory containing nonzero bins beyond its
query cutoff; {r['results']['published_input']['queries_attempting_same_or_future_papers']}
queries directly accessed a same-day or future-dated paper. Historical tracks
block those accesses or truncate the vectors before scoring. Temporal validity
still relies on the documented release-construction convention rather than
independent citation-event timestamps. New and repeat recall are secondary
outcomes; full-precision paired estimates accompany the supplement.
In the historical tracks, the raw-vector access counters identify records
whose original vectors contain unavailable bins; they do not mean that those
bins reached the scorer after truncation.

The audit counts unavailable bin endpoints, not independently confirmed future
citation events. Because bins are cumulative, summing a longer vector can repeat
earlier citations even without a new citation after the query. The three tracks
therefore diagnose input-policy sensitivity; their differences are not a causal
estimate of the benefit of future citation events.

All new comparisons are post-result supplementary analyses. No model was fitted
or selected using these outcomes, and no original primary result was replaced.
The newly located official score was not retrofitted into the ranker's features
or candidate generator; those retain the originally frozen approximation.
The published-input track is a provenance sensitivity, not evidence of a
historically deployable comparator. Python hash seed and interpreter were fixed
because upstream padding depends on hash and set iteration.

'''
    text=(STUDY/'MANUSCRIPT_DRAFT.md').read_text(encoding='utf-8')
    text=replace_once(text,'# Temporal Learning-to-Rank for Scientific Collaborator Discovery: A Reproducible Offline Study',
        '# Evaluating Ranking Objectives and Temporal Information in Scientific Collaborator Forecasting')
    text=re.sub(r'\*\*Author-review manuscript,.*?\*\*',
        '**Revised author-review manuscript, 8 November 2026. Includes an internal scientific review and supplementary official-source comparisons. Human author approval and public-availability arrangements remain required.**',text,count=1)
    abstract=f'''Scientific collaborator forecasting requires separating aggregate ranking gains from discovery of new partners. We evaluate library-based LambdaMART in a primary temporal collaborator task and two complementary graph and expertise tasks. On 1,000 PreScience queries, the practical ranker achieved nDCG@1000 of 0.4880 versus 0.4780 for a matched classifier. Adding text to network and recency features reduced performance in a fixed-candidate factorial, although the full pipeline improved over a separately retrained no-content pipeline. Median warm latency was 1.831 versus 1.146 seconds. Gains did not establish improved new-collaborator recall. A subsequent source audit located the official fixed scorer on a non-default repository branch and identified untruncated citation-trajectory use. With historical masking and harmonized inputs, that scorer achieved nDCG@1000 of {safe['ndcg1000']:.4f}; the frozen ranker's paired difference was {c['delta']:+.4f} (95% interval {c['ci95'][0]:+.4f} to {c['ci95'][1]:+.4f}). This comparison is supplementary and post-result. On ogbl-collab, the ranker underperformed its classifier and Neural Common Neighbor; on adapted exHarmony, it did not establish superiority over BM25. Findings support reporting information access, candidate construction, ranking objectives and practical retrieval outcomes separately.'''
    if len(abstract.split())>200:raise ValueError('Abstract exceeds 200 words')
    text=re.sub(r'(?<=## Abstract\n\n).*?(?=\n\nKeywords:)',abstract,text,count=1,flags=re.S)
    text=replace_once(text,'The intended contribution is an experimentally supported account of these questions under explicit information and computational constraints. The use of multiple agents or an existing ranking library is not itself a methodological contribution.',
        'The contribution is an empirical account of candidate-policy and objective-dependent effects, together with an executable temporal audit of a strong published comparator. PreScience is the primary collaborator-forecasting experiment; ogbl-collab and adapted exHarmony test whether the ranking-objective pattern transfers to different scholarly tasks. They are not three replications of a single matchmaking task or three tests of the same text/network/time model. The study introduces no new ranking algorithm, and does not treat multiple-agent orchestration or use of a ranking library as novelty.')
    text=text.replace('Generated proposals and outreach messages are outside the evaluation scope.',
        'The evaluated benchmark code is a separate offline research extension motivated by the matching module; it is not a fidelity test of the original application implementation. Generated proposals, outreach messages and agent coordination are outside the evaluation scope.',1)
    text=replace_once(text,'The accessible abstract establishes substantial overlap in motivation; its full implementation and evaluation remain to be examined.',
        "Its publisher abstract and official repository establish overlap in motivation. The released pipeline uses filtered DBLP graphs, author-representation pretraining and a sequential transformer. Its dataset-specific checkpoints cannot be transferred directly to PreScience author IDs. We do not execute TEDRec or claim superiority over it [13].")
    text=replace_once(text,'Some numerical details in its published description remain unspecified, so the current implementation is labeled approximate.',
        'The original experiment used an explicitly approximate prose reconstruction. The present revision adds execution of the subsequently located official scorer and distinguishes its raw released inputs from historical-input adaptations [12].')
    text=replace_once(text,'Three expanded NCN development runs have completed; the completion extension has not been executed.',
        'We execute NCN under a disclosed local training protocol; its completion extension is not evaluated.')
    text=replace_once(text,"The small count difference from the latest paper's reported release remains unresolved and prevents a claim of exact release reproduction.",
        'This count matches the official dataset card, which also lists 464,942 records in the test companion corpus. The paper reports nine fewer training records; the exact correspondence to that unspecified paper snapshot is unknown. This is a documentation/snapshot discrepancy, not evidence of a corrupt local download.')
    task_table='''
| Evaluation | Prediction unit | Available inputs | Claim boundary |
|---|---|---|---|
| PreScience, primary | Prior-history author to later coauthors | Strictly earlier text, collaboration and recency | Retrospective coauthorship forecasting |
| ogbl-collab, complementary | Future author-pair edge | Fixed historical graph; separate released-vector track | Global edge ranking against official negatives |
| Adapted exHarmony, complementary | Manuscript query to experts | Query title/abstract and undated expert corpus | Authorship/citation-based expertise retrieval |

'''
    text=text.replace('## 4. Methods\n',task_table+'## 4. Methods\n',1)
    text=text.replace('Saved models are replayed to verify export equivalence.',
        'Saved models are replayed to verify export equivalence. The saved LambdaRank models retain truncation level 30, normalization enabled and sigmoid 1. Thus validation selection at nDCG@1000 does not mean that the training surrogate directly optimizes all 1,000 positions. Equal grid counts control search effort, not equal optimality for different model families.',1)
    text=text.replace('An audit of the available official frequency helper verifies only that component. It does not validate the whole composite scorer.',
        'An original audit of the official frequency helper verified only that component. The post-result source audit and execution in Section 5.3 now address the full official scorer; original approximation results remain separately labeled.',1)
    text=text.replace('The PreScience v2 comparator is a description-based reimplementation, not an exact official Autoresearch reproduction [3].',
        "The original frozen experiment's PreScience v2 comparator was a description-based reimplementation, not an exact official Autoresearch reproduction [3].",1)
    text=text.replace('Official test data have not been used to select configurations.',
        'The original learned configurations were selected without official test outcomes; later supplementary comparisons are identified separately.',1)
    text=text.replace('Final inference retains the graph through2017, without adding validation edges.',
        'Final inference retains the graph through2017, without adding validation edges. LambdaRank groups training candidates by source author, while official Hits@50 uses a global negative threshold; this objective/evaluation mismatch limits general conclusions about ranking losses.',1)
    # Reorder the primary experiment before complementary tasks and costs.
    start=text.index('### 5.2 ');end=text.index('## 6. Discussion')
    block=text[start:end]
    sections=re.split(r'(?=### 5\.[2345] )',block)
    parts={s.split('\n',1)[0].split()[1]:s for s in sections if s.strip()}
    if set(parts)!={'5.2','5.3','5.4','5.5'}:raise ValueError('Unexpected result section layout')
    replacement=parts['5.5'].replace('### 5.5','### 5.2',1)+extension+parts['5.2'].replace('### 5.2','### 5.4',1)+parts['5.3'].replace('### 5.3','### 5.5',1)+parts['5.4'].replace('### 5.4','### 5.6',1)
    text=text[:start]+replacement+text[end:]
    text=re.sub(r'Table\s*[123] reports', 'The following table reports', text)
    text=text.replace('The unresolved PreScience training-row discrepancy, approximate fixed scorer, adapted exHarmony split and node-vector provenance prevent claims of exact published-benchmark reproduction.',
        'The paper-versus-data-card snapshot discrepancy, original approximate comparator, supplementary input adaptations, adapted exHarmony split and unverified node-vector provenance prevent claims of exact published-benchmark reproduction.',1)
    dk=list(dep['results']);da=dep['results'][dk[0]]['shared_observed_team_components'];db=dep['results'][dk[1]]['shared_observed_team_components']
    robustness=f'''A supplementary dependence analysis joined queries sharing any observed target-team author into {dep['components']} components (largest {dep['largest_component']} queries). The ranker-minus-classifier difference remained {effect(da)}, and the full-factorial-minus-network+recency difference remained {effect(db)}. Their descriptive signs were unchanged across all four calendar quarters. This diagnostic addresses additional observed-team overlap but not all shared-candidate or historical-network dependence; it is post-result and not independent confirmation.

'''
    text=text.replace('## 6. Discussion — interpretation boundaries\n\n','## 6. Discussion — interpretation boundaries\n\n'+robustness+verdict+' The strongest defensible conclusion remains conditional pipeline behavior, not universal superiority.\n\n',1)
    text=text.replace('## 7. Reproducibility and availability',
        'Absence of observed coauthorship is not evidence that two researchers would be incompatible. The retrospective labels and sampled negative examples can miss plausible collaborations that did not occur or were not captured in the release. Eligibility excludes authors without earlier publications, and selecting the first eligible author does not represent every possible query-author policy. Queries are conditioned on observed publication events; the experiment does not estimate whether or when a collaboration will occur. The sampled cohort supports conditional forecasting comparisons within this corpus, not population-wide utility or strict cold-start performance.\n\n## 7. Reproducibility and availability',1)
    text=text.replace('measured results are reported in Section5.4','measured results are reported in Section5.6',1)
    text=text.replace('The practical PreScience ranker improves\naggregate nDCG over its matched classifier and specified fixed comparators, but',
        'The practical PreScience ranker improves\naggregate nDCG over its matched classifier and original approximate comparators, but',1)
    text=text.replace('## Declarations for author completion',
        'The supplementary official-source comparison further constrains baseline claims: '+verdict+'\n\n## Declarations for author completion',1)
    conclusion=f'''## 8. Conclusion

The revised evaluation does not establish general superiority of the practical
LambdaMART pipeline. On the same 1,000 PreScience queries, the official fixed
scorer with harmonized historical inputs achieved nDCG@1000 of {safe['ndcg1000']:.4f},
versus {full['ndcg1000']:.4f} for the frozen ranker. {persistence}
The positive matched-classifier comparison remains a conditional result about
the evaluated objective and features, not evidence that learning to rank is the
strongest available pipeline. Text, network and explicit-recency effects depend
on the candidate policy and feature combination; adding text in the fixed-pool
factorial reduced performance, and aggregate gains did not establish improved
new-collaborator discovery. Complementary graph and expertise tasks also failed
to support universal superiority. Executable comparator provenance, historical
information access, candidate construction, uncertainty and practical retrieval
outcomes should therefore be reported separately. The study evaluates an offline
ranking extension and does not establish real-world collaboration success or
the efficacy of the eight-agent application.

'''
    text=re.sub(r'## 8\. Conclusion.*?(?=## Declarations for author completion)',conclusion,text,count=1,flags=re.S)
    text+='\n12. PreScience authors. Official Autoresearch collaborator scorer and citation-metadata construction code, commit e54f6a5441aa389c18acfc11080facea6a95c1d5. https://github.com/allenai/prescience/blob/e54f6a5441aa389c18acfc11080facea6a95c1d5/task_coauthor_prediction/baseline_autoresearch.py (accessed 7 November 2026).\n'
    text+='13. Guan, K. et al. TEDRec official implementation, commit e02f059ac98924267fc52059d8f0f4f3f4d00aad. https://github.com/YangSuoly/TEDRec (accessed 8 November 2026).\n'
    text=text.replace('### 5.3 Supplementary audit',
        '![Figure 1. Original frozen fixed-pool factorial. Points are three-seed means; error bars are model-seed standard deviations, not confidence intervals. History controls and the full candidate pool remain in every cell; composite scores are excluded.](figures/factorial_original.png)\n\n### 5.3 Supplementary audit',1)
    text=text.replace('### 5.4 Temporal graph prediction',
        '![Figure 2. Supplementary official-source comparisons. Positive differences favor the frozen ranker; intervals are marginal 95% author-cluster intervals. The three new tests form a separate Holm-adjusted family and are post-result.](figures/official_comparator_contrasts.pdf)\n\n### 5.4 Temporal graph prediction',1)
    (DEST/'MANUSCRIPT_REVISED.md').write_text(text,encoding='utf-8')
    main_text=text[text.index('## 1. Introduction'):text.index('## Declarations for author completion')]
    main_text=re.sub(r'## 4\. Methods.*?(?=## 5\. Results)','',main_text,flags=re.S)
    format_audit=dict(title_words=len(text.splitlines()[0].lstrip('# ').split()),abstract_words=len(abstract.split()),
        main_text_words_including_tables=len(main_text.split()),
        basis='Scientific Reports submission guidance checked 8 November 2026; count excludes abstract, Methods, declarations and references, and conservatively includes table text and Markdown figure captions',
        source='https://www.nature.com/srep/author-instructions/submission-guidelines',
        limits=dict(title=20,abstract=200,recommended_main_text=4500),
        author_declarations='pending named human authors; not fabricated')
    if format_audit['title_words']>20:raise ValueError('Title too long')
    (DEST/'format_audit.json').write_text(json.dumps(format_audit,indent=2),encoding='utf-8')
    supp=(STUDY/'SUPPLEMENTARY_INFORMATION.md').read_text(encoding='utf-8')
    supp=supp.replace('Author-review version, 29 October 2026.','Revised author-review version, 8 November 2026. Original frozen results are preserved; post-result additions are explicitly separated.',1)
    supp=supp.replace("The exact final eligibility counts\nand dates must be taken from the completed run's data audit.",
        'The frozen sample includes 1,000 of 47,997 eligible targets and 989 seed-author clusters, covering four calendar quarters from October 2024 through September 2025.',1)
    supp=supp.replace('Archive read-back and final visual review must be recorded before delivery.',
        'The original archive passed full read-back and its PDFs passed visual review. The dated revision carries a separate delivery manifest.',1)
    supp=supp.replace('The PreScience v2 fixed scorer is reconstructed from its published description;',
        'The original PreScience v2 comparator was reconstructed from its published description;',1)
    supp=supp.replace('Final results\ncannot resolve the upstream implementation ambiguity or the nine-record\ntraining-release discrepancy.',
        'The original final results do not validate this approximation as the official implementation. Section S7 adds the pinned official scorer without replacing any original result. The official dataset card documents 373,716 training records, matching the local file; the paper text describes nine fewer records. Exact correspondence to that earlier paper snapshot remains unknown.',1)
    supp=supp.replace('in the global lock; final runners perform inference only.',
        'in the global lock; final runners perform inference only. Saved LambdaRank models use truncation level 30, normalization enabled and sigmoid 1. The selection metric nDCG@1000 is therefore distinct from the training surrogate. Equal grid counts do not establish equally optimal tuning.',1)
    supp+='\n## S7. Supplementary official-source comparison\n\n'+extension.split('\n',1)[1]
    supp+='\n## S8. Dependence and period sensitivity\n\n'+robustness
    supp+='| Contrast | Quarter | Queries | Mean nDCG difference |\n|---|---|---:|---:|\n'
    for name,v in dep['results'].items():
        short='Ranker minus classifier' if name.startswith('full_') else 'Full factorial minus network+time'
        for q,vv in sorted(v['quarters'].items()):supp+=f"| {short} | {q} | {vv['queries']} | {vv['delta']:+.6f} |\n"
    supp+='\n## S9. Reproduction and temporal boundary checks\n\nThe original global lock and all saved models remain unchanged. The supplementary-lock.json binds the original cohort, source files, scorer, corpus and interpreter; runtime-addendum.json binds the indexed helper. Original scorer/source AST parity is verified against the upstream Git blob. Seven tests cover unchanged-input parity, citation-bin boundaries, future-reference hiding, author deduplication, future-citation mutation, instrumentation and missing-relevance denominators. The indexed helper is checked against the upstream helper for its first 1,000 calls and every 1,000th later call. Independent pytrec_eval checks cover all supplementary queries/tracks. Checkpoint hashes and source manifests accompany the revised package.\n\nNew scorer run durations are operational, with input-adapter overhead and possible desktop contention. They are not used as a fair speed comparison with the isolated original warm-latency experiment. No model/configuration has been selected from the supplementary outcomes.\n'
    supp=supp.replace('runtime-addendum.json binds the indexed helper.',
        'runtime-addendum.json and runtime-bounded-addendum.json bind the indexed helpers and bounded LRU cache.')
    supp=supp.replace('Independent pytrec_eval checks cover all supplementary queries/tracks.',
        'Five additional indexed/bounded-helper tests verify missing/duplicate records, empty histories, same-day boundaries, reverse query order and cache eviction. Independent pytrec_eval checks cover all supplementary queries/tracks.')
    supp+='\nAn outer resource wrapper waits at the existing RAM reserve while retaining loaded data. Its separately hashed addendum and wait receipt document this scheduling adaptation. Neither resource waits nor operational priority changes are treated as scientific cost measurements.\n'
    supp=supp.replace('jaccard10','`jaccard10`')
    (DEST/'SUPPLEMENTARY_REVISED.md').write_text(supp,encoding='utf-8')
    report=f'''# ResearchForge AI

Scientific review and updated benchmark evidence

8 November 2026 | Prepared for supervisor review | Proposed target: Scientific Reports

## What changed after the first report

The original numerical study was complete, but an internal referee-style review identified an important missing comparator. The official PreScience fixed scorer was available on non-default repository branches. We located and pinned it, examined its information access, and executed three fixed input tracks on the original 1,000-query cohort. No earlier model or result was overwritten. This report supersedes the earlier report for interpretation of the strongest-baseline comparison.

## Main scientific message

Existing learning-to-rank tools can help some historical collaborator-ranking comparisons, but gains depend on the candidate policy, feature combination and comparator. Better aggregate ranking does not establish better first-time collaborator discovery. The original eight-agent prototype remains outside the empirical claim: this study evaluates a separate offline ranking extension motivated by its matching component.

## Completed evidence

| Setting | Evaluation size | Main finding |
|---|---:|---|
| PreScience | 1,000 queries; 989 seed authors | Ranker nDCG 0.488033 versus classifier 0.477962 |
| ogbl-collab | 46,329 positives; 100,000 shared negatives | Ranker Hits@50 0.524042 versus NCN 0.536295 |
| Adapted exHarmony | 1,589 Authors queries | Ranker nDCG@10 0.121470 versus BM25 0.126221; no established advantage |

These tasks use different labels and metrics. They must not be combined into a single accuracy score. OGB and exHarmony complement the primary collaborator study; they do not replicate the complete text/network/time pipeline.

## Official baseline: the decisive update

{table}

{verdict} The ranker-minus-harmonized-scorer difference is {effect(c)}. {persistence} The three-track family is a supplementary comparison designed after the original final scores were known, not an untouched confirmatory experiment.

The raw published-input track must not be interpreted as a leakage-controlled historical baseline. Its source consumes complete cumulative citation trajectories, which can contain later information. We separately evaluated historical prefixes and strict paper-date masks, then also aligned author cleaning with the study. No claim is made about undisclosed preprocessing behind the original paper's published numbers.

## What remains stable under the scientific audit

The original classifier comparison remains positive after grouping queries that share any observed team author: {effect(da)} across {dep['components']} components. The negative text-factorial contrast also remains: {effect(db)}. Both point-estimate signs persist in every calendar quarter. Shared historical-network dependence remains outside this diagnostic.

The fixed-pool factorial and end-to-end content ablation answer different questions. Adding text features to network+time reduces nDCG by0.009147. The complete content-enabled pipeline improves over its separately retrained no-content counterpart by0.012078, despite lower candidate recall. This does not identify a single causal mechanism.

## Practical meaning and cost

New-collaborator Recall@10 is only0.026523 for the original full ranker, versus0.028320 for its matched classifier. This is a limitation for discovering genuinely new partners. Benefits are clearer for repeat collaborations and richer publication histories.

The measured original median warm query times are1.831seconds with content and1.146seconds without it, on30development queries with three repeats. These exclude preprocessing and training, and are not end-to-end timings of all eight agents. The no-content tail latency was not uniformly lower.

## Referee-style conclusion

The initial recommendation was major revision before submission. The official-comparator omission, temporal-input audit, task framing, dependence sensitivity and model-default reporting have now been addressed in the revised materials. Negative and inconclusive outcomes remain visible. The evidence supports a bounded empirical paper; it does not establish a new algorithm or state-of-the-art performance.

Close approaches such as TEDRec and CoBERT were not executed on PreScience. TEDRec's public models are tied to a different dataset and author mapping, so an honest comparison would require separate adaptation and retraining. The article explicitly excludes superiority claims over those methods.

## Before submission

The named authors must review the evidence, decide whether the contribution is sufficient, approve the text and AI-use disclosure, and supply authorship, affiliations, funding and conflict statements. A permanent code/data access arrangement is still required. The AI-assisted internal review is not an independent professor's approval or an editorial decision. Scientific Reports acceptance is unknown.

'''
    report=report.replace('## What remains stable under the scientific audit',
        '![Figure 1. The three supplementary official-source comparisons. Positive differences favor the frozen ranker. Marginal intervals are separate from the three-contrast Holm correction; these analyses are post-result.](figures/official_comparator_contrasts.pdf)\n\n## What remains stable under the scientific audit',1)
    report=report.replace('## Practical meaning and cost',
        '![Figure 2. Original frozen component results. Points are means across three model seeds; error bars show seed standard deviation, not confidence intervals. All cells retain history controls and the full candidate pool.](figures/factorial_original.png)\n\n## Practical meaning and cost',1)
    checklist='''## Review action checklist

| Finding | Revision made | Remaining boundary |
|---|---|---|
| R1: comparator | Executed the pinned official scorer in three input tracks. | Not a replication of the paper's unpublished execution environment. |
| R2: time access | Audited citation bins and masked same-day/future papers. | Citation-event dates were not independently reconstructed. |
| R3: task framing | Made PreScience primary and distinguished the two complementary tasks. | No pooled accuracy or three identical replications. |
| R4: new partners | Reported weak first-time collaborator recall prominently. | No evidence about willingness or practical compatibility. |
| R5: ablations | Separated fixed-pool component effects from complete pipeline changes. | No causal mechanism identified. |
| R6: uncertainty | Added shared-team component resampling and quarter diagnostics. | Some network dependence remains. |
| R7: tuning | Disclosed LambdaRank truncation, objective mismatch and unequal search budgets. | Results concern the tested configurations. |
| R8: prior methods | Audited available TEDRec code and updated related work. | TEDRec and CoBERT were not retrained; no SOTA claim. |
| R9: presentation | Reordered results, removed stale wording and rebuilt the review documents. | Human scientific approval remains necessary. |
| R10: data counts | Matched downloaded counts to the official dataset card. | The paper's earlier nine-row snapshot difference remains unidentified. |
| R11: availability | Prepared reproducibility instructions, evidence and a dated extension archive. | Authors must approve declarations and arrange actual code/data access. |

'''
    report=report.replace('## Referee-style conclusion',checklist+'## Referee-style conclusion',1)
    (DEST/'PROFESSOR_REVIEW_REPORT.md').write_text(report,encoding='utf-8')
    ledger=f'''# Response to the internal scientific review

8 November 2026. This documents an AI-assisted internal revision, not external
peer review. The original recommendation was major revision. The local analyses
and manuscript corrections listed below are complete; human author approval and
an actual code/data access arrangement remain before submission.

| Finding | Resolution | Evidence / remaining boundary |
|---|---|---|
| R1: approximate strongest comparator | Executed pinned official source on the original 1,000 queries, under all three declared input tracks; retained the approximation separately. | official-comparator-results.json; source-parity.json; no test tuning. |
| R2: future information | Added strict paper-date and citation-bin input adapters; reported raw inputs as a provenance sensitivity. | Seven tests; access audit; release-builder bin convention, not independently reconstructed event times. |
| R3: unlike tasks | Reframed PreScience as primary and OGB/exHarmony as complementary; added task/input/claim table. | Revised manuscript Sections 1 and 3; no pooled accuracy or three identical replications. |
| R4: first-time collaborator claim | Put low new-collaborator recall and lack of established improvement in abstract, results and report. | New Recall@10 0.026523 versus classifier 0.028320; retrospective forecasting only. |
| R5: component attribution | Distinguished fixed-pool factorial from retrained, different-candidate no-content pipeline; retained negative text effect. | Revised Methods/Discussion and supplement S2; no causal mechanism claim. |
| R6: dependence | Added shared-observed-team component bootstrap and quarter diagnostics. | {dep['components']} components; largest {dep['largest_component']}; dependence-robustness.json. Does not cover all historical-network overlap. |
| R7: objective/tuning budget | Reported saved LambdaRank defaults and truncation 30; noted OGB per-source training versus global Hits@50 and unequal NCN selection budget. | Revised Methods and supplement S3. No retrospective tuning. |
| R8: close-method coverage | Examined TEDRec official code/release and updated related work; explicitly excluded superiority over unexecuted methods. | UPSTREAM_COMPARISON.md; code is available, but dataset-specific retraining is outside this bounded study. |
| R9: scientific presentation | Reordered primary results first; removed stale execution language; rebuilt report, manuscript and supplement. | Dated revision; rendering/visual receipts are separate. |
| R10: release mismatch | Verified current official card equals local 373,716 training / 464,942 test-corpus counts; documented nine-row paper snapshot discrepancy. | Local corruption is not indicated by this discrepancy; exact old-snapshot identity remains unknown. |
| R11: availability and human responsibility | Prepared source/prediction/protocol inventory and additive reproducibility package. | Public deposit, author declarations and author approval are still required external actions; none is fabricated or implied. |

## What the new comparator changes

{verdict} The harmonized historical fixed scorer reaches
{safe['ndcg1000']:.6f} nDCG@1000, versus {full['ndcg1000']:.6f} for the frozen
ranker; paired difference {effect(c)}. This is a post-result supplementary
comparison. Its three-test correction does not retroactively change the
original primary family.

## Completion boundary

The requested local referee-style review and official-source comparison are
resolved. This does not guarantee that a journal reviewer will request no further
experiments. In particular, close deep collaborator methods were not retrained,
historical citation provenance relies on the released builder, and no prospective
or human utility evaluation was performed. The revised claims are bounded to the
evidence actually collected. Publication readiness still requires the named
authors' scientific judgment, declarations and a concrete access arrangement.
'''
    (HERE/'REVIEW_RESOLUTION.md').write_text(ledger,encoding='utf-8')
    for name in ['REFEREE_REPORT.md','REVIEW_RESOLUTION.md','UPSTREAM_COMPARISON.md','REVIEW_PROTOCOL.md','ADDITIONAL_REVIEW_ANALYSES.md','REPRODUCTION.md','official-comparator-results.json','dependence-robustness.json','source-parity.json','supplementary-lock.json','runtime-addendum.json','runtime-parity.json','runtime-bounded-addendum.json','runtime-bounded-parity.json','runtime-memory-wait-addendum.json','runtime-memory-wait.json','indexed-helper-tests.log','bounded-helper-tests.log']:
        shutil.copy2(HERE/name,DEST/name)
    shutil.copy2(HERE/'final-output-audit.json',DEST/'final-output-audit.json')
    files={p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.suffix in {'.md','.json','.log'}
           and p.name not in {'source_manifest.json','pdf_manifest.json','VISUAL_QA.json'}}
    manifest=dict(date='2026-11-08',original_manuscript_sha256=sha(STUDY/'MANUSCRIPT_DRAFT.md'),
        original_global_lock_sha256=sha(STUDY/'final/global-lock.json'),abstract_words=len(abstract.split()),files=files)
    (DEST/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(DEST),'abstract_words':len(abstract.split()),'official_nDCG':safe['ndcg1000'],'verdict':verdict}))

if __name__=='__main__':main()
