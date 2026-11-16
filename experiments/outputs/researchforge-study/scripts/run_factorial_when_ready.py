"""Single-writer dependency queue for the first real hybrid factorial refit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3];SCRIPTS=Path(__file__).resolve().parent
SOURCE=ROOT/'outputs/researchforge-study/results/hybrid-graph-dedup-200-100-1000-s42'
STATE=ROOT/'work/factorial-chain-state.json';LOCK=ROOT/'work/factorial-chain.lock'

def state(stage,**extra):
    value=dict(stage=stage,supervisor_pid=os.getpid(),updated_epoch=time.time(),**extra)
    atomic_state(STATE,value)

def main():
    fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid()),f)
    expected={n:hashlib.sha256((SCRIPTS/n).read_bytes()).hexdigest() for n in
              ['fit_factorial.py','publication_ablation_design.py','prescience_pilot.py']}
    try:
        while True:
            if (SOURCE/'factorial-s42/report.json').exists():state('already complete');return
            if not (SOURCE/'report.json').exists():
                parent=ROOT/'work/hybrid-chain-state.json'
                if parent.exists():
                    status=json.loads(parent.read_text())
                    if status['stage']=='failed' or status['stage'].startswith('encoder stopped'):
                        state('dependency requires repair',dependency=status);return
                state('waiting for completed hybrid development report');time.sleep(30);continue
            if psutil.virtual_memory().available<1800*1024**2 or shutil.disk_usage(ROOT).free<1024**3+150*1024**2:
                state('waiting for resources');time.sleep(30);continue
            for n,digest in expected.items():
                if hashlib.sha256((SCRIPTS/n).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Queued source changed; inspect before restarting')
            command=[sys.executable,'-u',str(SCRIPTS/'fit_factorial.py'),'--source',str(SOURCE),'--seed','42']
            with (ROOT/'work/factorial-pilot.log').open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state('running development factorial',worker_pid=child.pid,command=command,code_sha256=expected)
                code=child.wait()
            if code:raise RuntimeError(f'Factorial runner exited{code}; inspect log')
            state('complete',report=str(SOURCE/'factorial-s42/report.json'));return
    except Exception as error:
        state('failed',error=repr(error));raise
    finally:LOCK.unlink()

if __name__=='__main__':main()
