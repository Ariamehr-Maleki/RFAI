"""Sequential larger DEVELOPMENT runs; preserve sealed adapted test judgments.

Each seed uses1000 training/500 validation queries from the pre-frozen split.
Training size and validation size are fixed before these scores are observed.
Sampling variation is included in across-seed SD; not optimization-only variance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[3]
SCRIPTS=Path(__file__).resolve().parent
OUT=ROOT/'outputs/researchforge-study/results'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-queries',type=int,default=1000)
    ap.add_argument('--valid-queries',type=int,default=500);ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    args=ap.parse_args();run=OUT/f'exharmony-expanded-{args.train_queries}-{args.valid_queries}';run.mkdir(exist_ok=True)
    config=dict(**vars(args),candidates=1000,pilot_sha256=hashlib.sha256((SCRIPTS/'exharmony_bm25_pilot.py').read_bytes()).hexdigest(),
                test_policy='Adapted test qrels remain sealed; development only')
    cfg=run/'config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:cfg.write_text(json.dumps(config,indent=2))
    reports=[]
    for seed in args.seeds:
        folder=OUT/f'exharmony-bm25-{args.train_queries}-{args.valid_queries}-1000-s{seed}'
        if not (folder/'report.json').exists():
            progress=json.loads((ROOT/'work/indices/prescience-minilm/progress.json').read_text())
            remaining=(progress['total_rows']-progress['completed_rows'])*384*2
            assert shutil.disk_usage(ROOT).free>1024**3+remaining+100*1024**2,'Insufficient disk headroom for another development run'
            assert psutil.virtual_memory().available>2500*1024**2,'Insufficient available RAM'
            command=[sys.executable,'-u',str(SCRIPTS/'exharmony_bm25_pilot.py'),'--train-queries',str(args.train_queries),
                     '--valid-queries',str(args.valid_queries),'--candidates','1000','--seed',str(seed)]
            with (run/f'seed-{seed}.log').open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                (run/'state.json').write_text(json.dumps(dict(status='running',seed=seed,supervisor_pid=os.getpid(),worker_pid=process.pid,command=command),indent=2))
                assert process.wait()==0,f'Pilot seed{seed} failed: inspect log'
        subprocess.run([sys.executable,str(SCRIPTS/'verify_exharmony_metrics.py'),str(folder)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        reports.append(json.loads((folder/'report.json').read_text()))
        print(f'Completed and independently verified seed{seed}',flush=True)
    metrics={}
    for judgment, methods in reports[0]['metrics'].items():
        metrics[judgment]={}
        for method, values in methods.items():
            metrics[judgment][method]={}
            for key in values:
                nums=[r['metrics'][judgment][method][key] for r in reports]
                metrics[judgment][method][key]=dict(mean=float(np.mean(nums)),sample_sd=float(np.std(nums,ddof=1)) if len(nums)>1 else None,per_seed=nums)
    result=dict(status='Expanded DEVELOPMENT; validation selected; no final test or confirmatory superiority',config=config,metrics=metrics,
                limitations=['Seeds change sampled queries and training sets; samples can overlap.',
                             'Across-seed SD is not a confidence interval or independent test replication.',
                             'Missing-author raw relevance judgments retained; Authors-trained transfer to Cite/SimCite.',
                             'No human evaluation; observed expert-proxy outcomes only.'])
    (run/'report.json').write_text(json.dumps(result,indent=2))
    (run/'state.json').write_text(json.dumps(dict(status='complete',supervisor_pid=os.getpid(),seeds=args.seeds)))
    print(json.dumps({j:{m:v['ndcg10'] for m,v in ms.items()} for j,ms in metrics.items()},indent=2))


if __name__=='__main__':main()
