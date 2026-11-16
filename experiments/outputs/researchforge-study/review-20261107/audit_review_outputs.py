"""Cross-check completed supplementary outputs against the original frozen cohort."""
import json
import math
import numpy as np
import official_comparator as base

def main():
    report=base.read(base.HERE/'official-comparator-results.json')
    lock=base.read(base.HERE/'supplementary-lock.json')
    if not report['status'].startswith('Complete') or report['config']!=lock:raise ValueError('Incomplete or mismatched report')
    if report['independent_metric_max_error']>1e-12:raise ValueError('Independent evaluation mismatch')
    baseline=base.baseline_rows();ids=set(baseline[42])
    fields=['seed','date','truth','new_truth','seed_history_count']
    for seed in [43,44]:
        for q in ids:
            for k in fields:
                if baseline[seed][q][k]!=baseline[42][q][k]:raise ValueError('Frozen-seed metadata mismatch')
    hashes={};maximum_error=0.
    for track in base.TRACKS:
        p=base.HERE/(track+'.predictions.json');rows=base.read(p)
        if len(rows)!=1000 or {r['qid'] for r in rows}!=ids:raise ValueError('Wrong evaluation cohort')
        for row in rows:
            q=row['qid'];original=baseline[42][q]
            if any(row[k]!=original[k] for k in fields):raise ValueError('Comparator labels or metadata differ: '+q)
            if row['config_sha256']!=base.sha(base.HERE/'supplementary-lock.json'):raise ValueError('Wrong checkpoint lock')
            pred=row['predicted'];scores=row['scores']
            if len(pred)!=1000 or len(set(pred))!=1000 or len(scores)!=1000 or row['seed'] in pred:raise ValueError('Invalid candidates')
            if not all(math.isfinite(x) for x in scores) or any(b>a for a,b in zip(scores,scores[1:])):raise ValueError('Invalid score order')
            measured=base.metrics(pred,row['truth'],row['new_truth'])
            for k,v in measured.items():
                if v is None:
                    if row[k] is not None:raise ValueError('Metric eligibility mismatch')
                else:maximum_error=max(maximum_error,abs(row[k]-v))
            checkpoint=base.HERE/'checkpoints'/track/(q+'.json')
            if base.read(checkpoint)!=row:raise ValueError('Combined file differs from checkpoint')
            hashes[str(checkpoint.relative_to(base.HERE)).replace('\\','/')]=base.sha(checkpoint)
        for metric,value in report['results'][track]['metrics'].items():
            measured=float(np.mean([r[metric] for r in rows if r[metric] is not None]))
            maximum_error=max(maximum_error,abs(measured-value))
        hashes[p.name]=base.sha(p)
    if maximum_error>1e-12:raise ValueError('Output/summary metric mismatch')
    result=dict(status='passed',queries_per_track=1000,tracks=base.TRACKS,rows=3000,
        original_seed_metadata_identical=True,checkpoint_rows_identical=True,
        maximum_recomputed_metric_error=maximum_error,
        independent_pytrec_eval_max_error=report['independent_metric_max_error'],
        report_sha256=base.sha(base.HERE/'official-comparator-results.json'),
        supplementary_lock_sha256=base.sha(base.HERE/'supplementary-lock.json'),
        statistics_source_sha256=base.sha(base.STUDY/'scripts/final_statistics.py'),
        output_sha256=hashes)
    base.write(base.HERE/'final-output-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='output_sha256'}),flush=True)

if __name__=='__main__':main()
