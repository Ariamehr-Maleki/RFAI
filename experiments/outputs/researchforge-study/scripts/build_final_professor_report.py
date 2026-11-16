"""Prepare an English review report only after all frozen results and figures exist.

Produces portable HTML, CSV tables, source provenance and copies of the current
manuscript/supplement. PDF rendering and visual review are separate steps.
"""
import argparse
import base64
import html
import json
from pathlib import Path
from final_evaluation_gate import verify_lock
from final_results_tables import checked_report, table_rows, csv_text, STATUSES
from frozen_ranker import digest
from supervisor_state import atomic_state

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'outputs/researchforge-study'
FIGURES = ['figure1_primary_contrasts', 'figure2_factorial',
           'figure3_secondary_subgroups', 'figure4_pipeline_cost']


def prerequisites(root):
    study = root / 'outputs/researchforge-study'
    required = [study / 'final/global-lock.json', study / 'MANUSCRIPT_DRAFT.md',
                study / 'SUPPLEMENTARY_INFORMATION.md']
    required += [study / 'results/final' / d / 'report.json' for d in STATUSES]
    required += [study / 'results/final' / d / 'provenance.json' for d in ['tables', 'figures']]
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError('Final professor report requires completed evidence: ' + ', '.join(missing))


def table(headers, rows):
    head = ''.join('<th>' + html.escape(str(v)) + '</th>' for v in headers)
    body = ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows)
    return '<div class="table-wrap"><table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>'


def label(value):
    return value.replace('__minus__', ' minus ').replace('__', ' / ').replace('_', ' ').replace('+', ' + ')


def verify_export_sources(root, provenance, name, report_sources):
    """Reject stale exports even when their own file hashes remain consistent."""
    if name == 'tables':
        records = provenance['source_reports']
        if set(records) != set(STATUSES):
            raise ValueError('Table provenance must include all three reports')
        declared = {v['path']: v['sha256'] for v in records.values()}
    else:
        declared = provenance['source_sha256']
    resolved = {}
    for relative, checksum in declared.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or digest(path) != checksum:
            raise ValueError('Export source differs: ' + relative)
        resolved[path] = checksum
    for relative, checksum in report_sources.items():
        if resolved.get((root / relative).resolve()) != checksum:
            raise ValueError('Export does not bind current report: ' + relative)


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError('Use a new output directory within the task workspace')
    prerequisites(root)
    study = root / 'outputs/researchforge-study'
    lock = verify_lock(root, study / 'final/global-lock.json')
    reports, sources = {}, {}
    for dataset in STATUSES:
        path, reports[dataset] = checked_report(root, lock, dataset)
        sources[str(path.relative_to(root))] = digest(path)
    report_sources = dict(sources)
    for name, expected in [('tables', 'Verified frozen-result table export'), ('figures', 'Verified frozen-result figures')]:
        folder = study / 'results/final' / name
        provenance = json.loads((folder / 'provenance.json').read_text())
        if provenance['status'] != expected or provenance['lock_sha256'] != lock['lock_sha256']:
            raise ValueError('Wrong export identity: ' + name)
        verify_export_sources(root, provenance, name, report_sources)
        sources[str((folder / 'provenance.json').relative_to(root))] = digest(folder / 'provenance.json')
        for relative, checksum in provenance['files'].items():
            path = (folder / relative).resolve()
            if not path.is_relative_to(folder.resolve()) or digest(path) != checksum:
                raise ValueError('Export artifact differs: ' + relative)
            sources[str(path.relative_to(root))] = checksum
    performance, contrasts = table_rows(reports)
    chunks = ['''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ResearchForge | Frozen benchmark study</title><style>
:root{--ink:#173047;--accent:#087f8c;--line:#dce5eb;--muted:#516778}
*{box-sizing:border-box}body{margin:0;background:#eef3f6;color:var(--ink);font:16px/1.6 Arial,sans-serif}
main{max-width:1080px;margin:auto;background:white;padding:56px 64px}header{border-top:8px solid var(--accent);padding-top:24px}
.eyebrow{color:var(--accent);font-weight:bold;letter-spacing:.12em;font-size:12px}h1{font-size:46px;line-height:1.1;margin:15px 0}
h2{font-size:27px;border-bottom:1px solid var(--line);padding-bottom:10px;margin-top:44px}h3{margin-top:28px}
.deck{font-size:20px;color:var(--muted)}.note{padding:18px 22px;background:#f1f7f8;border-left:4px solid var(--accent)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th{text-align:left;background:#173047;color:white}
td,th{padding:10px 12px;vertical-align:top;border-bottom:1px solid var(--line)}tr:nth-child(even){background:#f4f7fa}
.table-wrap{overflow-x:auto}figure{margin:30px 0;break-inside:avoid}img{max-width:100%;height:auto}figcaption{font-size:13px;color:var(--muted)}
code{overflow-wrap:anywhere;font-size:11px}a{color:#087f8c}footer{font-size:12px;margin-top:40px;color:var(--muted)}
@media print{body{background:white}main{max-width:none;padding:0}h1{font-size:32px}h2{break-after:avoid}table{font-size:9px}td,th{padding:6px}thead{display:table-header-group}tr{break-inside:avoid}.table-wrap{overflow:visible}.page{break-before:page}@page{size:A4;margin:17mm}}
</style><main><header><div class="eyebrow">RESEARCHFORGE AI · RESEARCH REVIEW</div>
<h1>Scientific collaborator ranking</h1><p class="deck">A reproducible study of content, collaboration networks and explicit time features across three held-out tasks.</p></header>
<p class="note">All numerical results below come from the globally frozen evaluations. This is an author/supervisor review report, not evidence of journal acceptance. The companion manuscript requires author review and declarations before submission.</p>
<h2>1. What was built and evaluated</h2>
<p>The original eight-agent prototype motivated the application. The study isolates its matching problem into a deterministic candidate-retrieval and ranking pipeline. Historical publication text, network evidence and recency features feed library-based LambdaMART, matched classifiers and fixed baselines. Proposal generation, outreach and agent coordination are outside the evaluated treatment.</p>
<p><strong>Flow:</strong> historical records → candidate retrieval → text/network/time features → frozen ranker → ordered researchers → comparison with recorded relevance. Target outcomes never enter prediction features; no human evaluation or LLM judge is used.</p>
<h2>2. Evaluation design</h2>''']
    p, o, e = (reports[k] for k in ['prescience', 'ogbl-collab', 'exHarmony'])
    full = p['seed_summary']['full_lambdamart']['ndcg1000']['mean']
    matched = p['seed_summary']['full_binary_lightgbm']['ndcg1000']['mean']
    chunks.append('<p class="note"><strong>Key finding.</strong> On the main collaborator task, the practical ranker scored '
                  + f'{full:.4f} versus {matched:.4f}'
                  + ' for its matched classifier. However, adding text to network and recency features in a fixed candidate pool reduced quality. The complete content-enabled pipeline helped relative to its no-content counterpart, while aggregate candidate recall fell. These are different interventions. New-collaborator recall remains low. On the graph task the ranker lost to NCN and its classifier; on expert retrieval it beat its classifier but did not establish an advantage over BM25.</p>')
    chunks.append(table(['Dataset', 'Frozen evaluation unit', 'Primary metric', 'Interpretation'], [
        ['PreScience', f"{p['queries']:,} sampled future-paper queries", 'nDCG@1000', 'Observed collaborator forecasting'],
        ['ogbl-collab', f"{o['positive_count']:,} positive edges; {o['negative_count']:,} shared negatives", 'Hits@50', 'Temporal graph prediction'],
        ['exHarmony', f"{e['queries']:,} adapted held-out queries", 'Authors nDCG@10', 'Expert retrieval, not collaboration outcomes']]))
    chunks.append('<p>Three model seeds are evaluated. Configuration selection used development data. The final family contains eight PreScience, three OGB and two exHarmony primary contrasts, with Holm correction within each dataset. Query/edge effects are averaged across model seeds before cluster inference. Marginal confidence intervals and model-seed SD answer different questions. No pooled accuracy is calculated across datasets.</p><h2>3. Final benchmark results</h2>')
    for dataset, judgment, metric in [('prescience', 'coauthors', 'ndcg1000'), ('ogbl-collab', 'edge', 'hits50'), ('exHarmony', 'authors', 'ndcg10')]:
        chunks.append('<h3>' + html.escape(dataset) + '</h3>')
        rows = [r for r in performance if r['dataset'] == dataset and r['judgment'] == judgment and r['metric'] == metric]
        chunks.append(table(['Method', metric, 'Model-seed SD'], [[label(r['method']), f"{r['mean']:.6f}", f"{r['sample_sd']:.6f}"] for r in rows]))
    chunks.append('<h2>4. Predeclared paired differences</h2><p>Positive differences favor the first named method; negative differences favor its comparator. Intervals are marginal 95% cluster-bootstrap intervals. A nonsignificant difference does not establish equivalence. These tests do not remove shared-neighbor or shared-expert dependence.</p>')
    chunks.append(table(['Task / contrast', 'Difference', '95% interval', 'Holm p'], [
        [r['dataset'] + ': ' + label(r['comparison']), f"{r['delta']:+.6f}", f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]", f"{r['holm_p']:.5g}"] for r in contrasts]))
    chunks.append('<h2>5. Component effects, subgroups and cost</h2><p>The fixed-pool factorial separates seven content, nine network and two explicit temporal columns while retaining three controls. Two composite scores are excluded from all eight cells. The separately retrained end-to-end no-content pipeline also changes retrieval. Removing explicit time does not remove historical eligibility or recent-publication windows.</p>')
    captions = ['Primary paired contrasts; marginal intervals, separate task metrics.',
                'Eight fixed-pool primitive-feature cells; error bars are model-seed SD.',
                'Secondary history strata and new/repeat recall; same full ranking, eligible subsets.',
                'Warm development latency and final quality; timings and quality use different query sets.']
    for name, caption in zip(FIGURES, captions):
        path = study / 'results/final/figures' / (name + '.png')
        if str(path.relative_to(root)) not in sources:
            raise ValueError('Figure missing from verified export')
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        chunks.append('<figure><img alt="' + html.escape(caption) + '" src="data:image/png;base64,' + encoded + '"><figcaption>' + html.escape(caption) + '</figcaption></figure>')
    chunks.append('''<h2>6. Interpretation and limitations</h2>
<p>The results assess whether existing ranking tools help under particular candidate and information constraints. They do not establish novelty of text/network/time fusion. A gain over a matched classifier is distinct from beating a stronger fixed or graph comparator. All negative results and all declared component cells are retained.</p>
<p>Coauthorship is a proxy for collaboration; missing future links are not verified incompatibilities. The PreScience fixed scorer is a published-description approximation, and a nine-record release discrepancy remains unresolved. The exHarmony split is adapted and its corpus undated. NCN and tree models have unequal validation-selection budgets; released node-feature temporal provenance is unverified. Query subsampling, three model seeds, candidate recall and incomplete dependence modeling bound generalization.</p>
<h2>7. Reproducibility and manuscript handoff</h2>
<p>Saved predictions, independent metric checks, exact source/model/data hashes, checkpoint receipts and the fixed configuration support audit. Models are not refitted on final labels. Raw corpora remain attributed to their original providers. Local artifact availability must not be described as a public archive DOI.</p>
<p>The manuscript and supplementary information accompany this report. Final author names/order, affiliations, contributions, funding and competing interests require the authors. Scientific Reports is the intended target; no submission, public deposit, external message or fee payment has been performed.</p>''')
    chunks.append('<footer>Global lock: <code>' + lock['lock_sha256'] + '</code><br>Full-precision numerical tables and source provenance accompany this report. PDF rendering and visual review must be recorded separately.</footer></main></html>')
    output.mkdir(parents=True)
    (output / 'ResearchForge_Final_Professor_Report.html').write_text('\n'.join(chunks), encoding='utf-8')
    (output / 'benchmark_results.csv').write_text(csv_text(performance), encoding='utf-8')
    (output / 'primary_contrasts.csv').write_text(csv_text(contrasts), encoding='utf-8')
    for filename in ['MANUSCRIPT_DRAFT.md', 'SUPPLEMENTARY_INFORMATION.md']:
        source = study / filename
        sources[str(source.relative_to(root))] = digest(source)
        (output / filename).write_bytes(source.read_bytes())
    atomic_state(output / 'source_manifest.json', dict(status='Verified evidence report; visual review and author approval pending',
        lock_sha256=lock['lock_sha256'], source_sha256=sources, builder_sha256=digest(__file__),
        files={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build(ROOT, args.output))
