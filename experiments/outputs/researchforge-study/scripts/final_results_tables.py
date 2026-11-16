"""Export manuscript tables ONLY from verified globally frozen final reports."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from final_evaluation_gate import verify_lock,canonical
from frozen_ranker import digest
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3]
STATUSES={'prescience':'Completed frozen PreScience evaluation','ogbl-collab':'Completed frozen ogbl-collab evaluation','exHarmony':'Completed frozen adapted exHarmony evaluation'}


def checked_report(root,lock,dataset):
    cfg=lock['payload']['plan']['datasets'][dataset];folder=(root/cfg['output_directory']).resolve()
    if not folder.is_relative_to(root):raise ValueError('Result path outside workspace')
    path=folder/'report.json';report=json.loads(path.read_text())
    if report['status']!=STATUSES[dataset] or report['lock_sha256']!=lock['lock_sha256'] or report['config_sha256']!=hashlib.sha256(canonical(cfg)).hexdigest():
        raise ValueError('Wrong final report identity: '+dataset)
    if report['config']!=cfg:raise ValueError('Report configuration differs')
    for name,sha in report['output_sha256'].items():
        target=(folder/name).resolve()
        if not target.is_relative_to(folder) or digest(target)!=sha:raise ValueError('Changed final output: '+name)
    return path,report


def table_rows(reports):
    performance=[];contrasts=[]
    for dataset,report in reports.items():
        if dataset=='ogbl-collab':
            for method,value in report['seed_summary'].items():performance.append(dict(dataset=dataset,judgment='edge',method=method,metric='hits50',mean=value['mean'],sample_sd=value['sample_sd']))
        elif dataset=='prescience':
            for method,metrics in report['seed_summary'].items():
                for metric,value in metrics.items():
                    if value is not None:performance.append(dict(dataset=dataset,judgment='coauthors',method=method,metric=metric,mean=value['mean'],sample_sd=value['sample_sd']))
        else:
            for judgment,methods in report['seed_summary'].items():
                for method,metrics in methods.items():
                    for metric,value in metrics.items():performance.append(dict(dataset=dataset,judgment=judgment,method=method,metric=metric,mean=value['mean'],sample_sd=value['sample_sd']))
        for name,value in report['primary_comparisons'].items():
            contrasts.append(dict(dataset=dataset,comparison=name,delta=value['delta'],ci95_low=value['ci95'][0],ci95_high=value['ci95'][1],holm_p=value['holm_p'],queries=value['queries'],clusters=value['clusters']))
    return performance,contrasts


def csv_text(rows):
    if not rows:raise ValueError('Empty manuscript table')
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);return stream.getvalue()


def export(root,lock_path,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if not output.is_relative_to(root) or output.exists():raise ValueError('New task-workspace export directory required')
    lock=verify_lock(root,lock_path);reports={};sources={}
    for dataset in STATUSES:
        path,report=checked_report(root,lock,dataset);reports[dataset]=report;sources[dataset]=dict(path=str(path.relative_to(root)),sha256=digest(path))
    performance,contrasts=table_rows(reports)
    lines=['# Frozen held-out results','',f"Global lock SHA: `{lock['lock_sha256']}`",'',
        'Each dataset measures a different task; no pooled accuracy is calculated. SD describes model-seed variation, not uncertainty of the mean. Primary contrasts use seed-averaged cluster inference and separate frozen Holm families.','']
    for dataset,metric,judgment in [('prescience','ndcg1000','coauthors'),('ogbl-collab','hits50','edge'),('exHarmony','ndcg10','authors')]:
        lines += [f'## {dataset}: {metric} ({judgment})','','| Method | Mean | Sample SD |','|---|---:|---:|']
        for row in performance:
            if row['dataset']==dataset and row['metric']==metric and row['judgment']==judgment:lines.append(f"| {row['method']} | {row['mean']:.6f} | {row['sample_sd']:.6f} |")
        lines+=['']
    lines += ['## Predeclared primary differences','','| Dataset | Comparison | Difference | 95% interval | Holm p | Queries / clusters |','|---|---|---:|---|---:|---:|']
    for row in contrasts:lines.append(f"| {row['dataset']} | {row['comparison']} | {row['delta']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['holm_p']:.6g} | {row['queries']} / {row['clusters']} |")
    lines += ['','Intervals are marginal. These tables preserve negative differences; no winner labels are generated. Secondary judgments and metrics are retained in the complete CSV.','']
    output.mkdir(parents=True)
    (output/'performance.csv').write_text(csv_text(performance),encoding='utf-8')
    (output/'primary_contrasts.csv').write_text(csv_text(contrasts),encoding='utf-8')
    (output/'RESULTS_TABLES.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_state(output/'provenance.json',dict(status='Verified frozen-result table export',lock_sha256=lock['lock_sha256'],source_reports=sources,
        export_code_sha256=digest(__file__),files={name:digest(output/name) for name in ['performance.csv','primary_contrasts.csv','RESULTS_TABLES.md']}))
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lock',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    print(export(ROOT,args.lock,args.output))
