"""Bounded local dependency chain: verified corpus stats -> pilot -> metric audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[3]
SCRIPTS=Path(__file__).parent
STATE=ROOT/'work/bm25-chain-state.json'

def save(**state):
    state.update(supervisor_pid=os.getpid(),updated_unix=time.time())
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATE)
    print(json.dumps(state),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stats-pid',type=int,required=True);args=ap.parse_args()
    expected=ROOT/'work/indices/exharmony-hash18/bm25_stats.json'
    save(stage='waiting_for_verified_stats',stats_pid=args.stats_pid)
    deadline=time.monotonic()+3600
    while not expected.exists():
        if not psutil.pid_exists(args.stats_pid):raise RuntimeError('Stats worker exited without completed manifest')
        if time.monotonic()>deadline:raise TimeoutError('Stats wait exceeded1hour')
        time.sleep(15)
    if psutil.virtual_memory().available<2300*1024**2:raise MemoryError('Need2.3GiB available for sparse conversion; retry later')
    run=ROOT/'outputs/researchforge-study/results/exharmony-bm25-200-100-1000-s42'
    commands=[('pilot',[sys.executable,'-u',str(SCRIPTS/'exharmony_bm25_pilot.py'),
              '--train-queries','200','--valid-queries','100','--candidates','1000']),
              ('metric_verification',[sys.executable,str(SCRIPTS/'verify_exharmony_metrics.py'),str(run)])]
    for stage,command in commands:
        log=ROOT/'work'/f'exharmony-bm25-{stage}.log'
        with log.open('w') as out:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
            save(stage=stage,child_pid=child.pid,command=command,log=str(log))
            result=child.wait()
        if result:raise RuntimeError(f'{stage} failed with exit{result}; see{log}')
    save(stage='complete',run=str(run))

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(stage='failed',error=repr(error));raise
