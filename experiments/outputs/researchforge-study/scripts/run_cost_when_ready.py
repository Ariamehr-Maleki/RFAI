"""One isolated development cost run after encoding/development queues finish."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3];SCRIPTS=Path(__file__).resolve().parent
STATE=ROOT/'work/cost-chain-state.json';LOCK=ROOT/'work/cost-chain.lock'
BASE=ROOT/'outputs/researchforge-study/results/prescience-expanded-1200-1000-500-1000'
OUTPUT=BASE/'isolated_pipeline_cost.json'
FILES=['pipeline_cost.py','reranking_cost.py','frozen_ranker.py','prescience_hybrid_graph.py','prescience_retrieval_ablation.py',
       'prescience_pilot.py','semantic_index.py','text_index.py','supervisor_state.py']


def state(stage,**extra):atomic_state(STATE,dict(stage=stage,supervisor_pid=os.getpid(),updated_epoch=time.time(),**extra))


def main():
    fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid()),stream)
    hashes={name:hashlib.sha256((SCRIPTS/name).read_bytes()).hexdigest() for name in FILES}
    try:
        while not OUTPUT.exists():
            if not (BASE/'report.json').exists() or not (ROOT/'outputs/researchforge-study/data-audit/final_semantic_index.json').exists():
                state('waiting for expanded development and completed label-free final encoding');time.sleep(30);continue
            if psutil.virtual_memory().available<2500*1024**2:state('waiting for2500MiB cost RAM after exact-value loader compaction');time.sleep(30);continue
            if any(hashlib.sha256((SCRIPTS/name).read_bytes()).hexdigest()!=sha for name,sha in hashes.items()):raise ValueError('Pinned cost source changed; inspect before restart')
            from reranking_cost import require_idle
            try:require_idle()
            except RuntimeError as error:state('waiting for isolated cost conditions',reason=str(error));time.sleep(30);continue
            command=[sys.executable,'-u',str(SCRIPTS/'pipeline_cost.py'),'--base',str(BASE),'--output',str(OUTPUT),'--queries','30','--repeats','3']
            with (ROOT/'work/isolated-pipeline-cost.log').open('a') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state('running isolated development costs',worker_pid=child.pid,command=command,code_sha256=hashes)
                code=child.wait()
            if code:raise RuntimeError(f'Cost runner exited{code}; inspect log, do not substitute contended experiment timings')
        report=json.loads(OUTPUT.read_text())
        if report['status']!='Completed isolated development pipeline cost; not final quality evaluation':raise ValueError('Unexpected cost report')
        state('complete',report=str(OUTPUT))
    except Exception as error:state('failed',error=repr(error));raise
    finally:LOCK.unlink()


if __name__=='__main__':main()
