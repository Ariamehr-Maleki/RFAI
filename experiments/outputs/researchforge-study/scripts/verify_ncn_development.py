"""Independently check saved NCN validation predictions, without Torch or raw labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def check(folder):
    folder=Path(folder);report_path=folder/'report.json';prediction_path=folder/'best.predictions.npz'
    report=json.loads(report_path.read_text())
    if 'DEVELOPMENT' not in report['status']:raise ValueError('Completed NCN development report required')
    with np.load(prediction_path,allow_pickle=False) as saved:
        positive=saved['positive'].reshape(-1);negative=saved['negative'].reshape(-1)
    if len(positive)!=60084 or len(negative)!=100000:raise ValueError('Unexpected full validation sizes')
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():raise ValueError('Nonfinite predictions')
    # OGB uses strict > against the kth-largest shared negative, not >=.
    threshold=float(np.partition(negative,-50)[-50]);hits=float(np.mean(positive>threshold))
    if abs(hits-report['best_validation_hits50'])>1e-12:raise ValueError('Saved predictions disagree with report')
    history=report['history'];epochs=[row['epoch'] for row in history]
    if epochs!=list(range(1,report['config']['epochs']+1)):raise ValueError('Incomplete or duplicated epoch history')
    best=max(history,key=lambda row:row['validation_hits50'])
    if abs(best['validation_hits50']-hits)>1e-12:raise ValueError('Saved checkpoint is not reported best validation epoch')
    result=dict(status='PASS',seed=report['config']['seed'],positive_count=len(positive),negative_count=len(negative),
                validation_hits50=hits,first_best_epoch=best['epoch'],completed_epochs=len(history),
                independent_threshold=threshold,tie_rule='strictly greater than50th-largest shared negative',
                sha256={p.name:digest(p) for p in [report_path,prediction_path]},
                limitations=['Validation-selected best of100epochs; not final test performance.',
                             'Verification covers saved predictions/score consistency, not complete input-pipeline correctness.'])
    (folder/'independent_metric_verification.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',type=Path)
    for run in parser.parse_args().runs:print(json.dumps(check(run)),flush=True)
