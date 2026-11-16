"""Sequential100-epoch temporal NCN development, same sampled data as tree models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SCRIPTS=Path(__file__).resolve().parent
OUT=ROOT/'outputs/researchforge-study/results/ncn-expanded-10000'


def main():
    OUT.mkdir(exist_ok=True);reports=[]
    for seed in [42,43,44]:
        run=OUT.parent/f'ncn-expanded-10000-e100-s{seed}'
        if not (run/'report.json').exists():
            command=[sys.executable,'-u',str(SCRIPTS/'ncn_temporal.py'),'--train-positives','10000','--epochs','100','--seed',str(seed)]
            with (OUT/f'seed-{seed}.log').open('a') as log:
                worker=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                (OUT/'state.json').write_text(json.dumps(dict(status='running',seed=seed,supervisor_pid=os.getpid(),worker_pid=worker.pid,command=command),indent=2))
                code=worker.wait()
                if code:
                    (OUT/'state.json').write_text(json.dumps(dict(status='stopped',seed=seed,exit_code=code,supervisor_pid=os.getpid())))
                    raise RuntimeError(f'NCN seed{seed} stopped with code{code}; inspect log and checkpoint')
        reports.append(json.loads((run/'report.json').read_text()))
        print(f'NCN seed{seed} complete',flush=True)
    values=[r['best_validation_hits50'] for r in reports]
    result=dict(status='Expanded DEVELOPMENT; validation-selected; official test sealed',seeds=[42,43,44],
                mean_hits50=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),per_seed=values,
                training_positive_count=10000,epochs=100,features='Released128-dimensional vectors; provenance caveat applies',
                limitations=['Not exact upstream training protocol: temporal2016 context and2017 labels,5 fixed negatives per positive.',
                             'Best of100 validation epochs, compared with3 tree configurations; selection budgets differ.',
                             'Across-seed variation includes training subsampling; no final test or superiority claim.'])
    (OUT/'report.json').write_text(json.dumps(result,indent=2));(OUT/'state.json').write_text(json.dumps(dict(status='complete',supervisor_pid=os.getpid())))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
