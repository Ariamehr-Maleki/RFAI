"""Label-free preprocessing after audited TRAIN encoding and available resources.

After exact-value loader compaction this can overlap one single-thread development
collector, using encoder threads2. It never overlaps isolated cost measurements.
"""
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
STATE=ROOT/'work/final-text-chain-state.json';LOCK=ROOT/'work/final-text-chain.lock'
DEST=ROOT/'work/indices/prescience-test-minilm'
DEPENDENCY=ROOT/'outputs/researchforge-study/data-audit/train_semantic_index.json'
FILES=['prescience_test_text_cache.py','semantic_encode.py','prescience_text_reuse_plan.py','supervisor_state.py']


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def state(stage,**extra):atomic_state(STATE,dict(stage=stage,supervisor_pid=os.getpid(),updated_epoch=time.time(),**extra))


def main():
    fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid()),stream)
    pinned={name:digest(SCRIPTS/name) for name in FILES}
    try:
        while not (DEST/'manifest.json').exists():
            if not DEPENDENCY.exists():state('waiting for audited TRAIN semantic cache; label-free preprocessing only');time.sleep(30);continue
            report=json.loads(DEPENDENCY.read_text())
            if not report['status'].startswith('PASS full TRAIN vector'):raise ValueError('Incomplete TRAIN audit')
            if not (ROOT/'work/indices/prescience-minilm/manifest.json').exists():raise ValueError('TRAIN manifest missing after audit')
            # Pinned b8/t2 encoder timing used565MB RSS, plus mapped reuse arrays.
            # Builder checks reserve after every committed batch.
            if psutil.virtual_memory().available<2500*1024**2 or shutil.disk_usage(ROOT).free<1024**3+400*1024**2:
                state('waiting for preprocessing RAM/disk');time.sleep(30);continue
            if any(digest(SCRIPTS/name)!=sha for name,sha in pinned.items()):raise ValueError('Queued preprocessing source changed; inspect before restart')
            command=[sys.executable,'-u',str(SCRIPTS/'prescience_test_text_cache.py')]
            with (ROOT/'work/final-text-cache.log').open('a') as stream:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                state('building label-free final semantic cache',worker_pid=child.pid,command=command,code_sha256=pinned)
                code=child.wait()
            if code:raise RuntimeError(f'Final text builder exited{code}; inspect resumable checkpoint/log')
        import numpy as np
        manifest=json.loads((DEST/'manifest.json').read_text());vectors=DEST/'embeddings.f16'
        if manifest['status']!='complete' or manifest['completed_rows']!=manifest['total_rows']:raise ValueError('Incomplete cache manifest')
        if vectors.stat().st_size!=manifest['bytes'] or digest(vectors)!=manifest['sha256']:raise ValueError('Final semantic SHA mismatch')
        values=np.memmap(vectors,mode='r',dtype=np.float16,shape=(manifest['total_rows'],384));norm_error=0.
        for start in range(0,len(values),4096):
            chunk=values[start:start+4096].astype(np.float32)
            if not np.isfinite(chunk).all():raise ValueError('Nonfinite completed vectors')
            norm_error=max(norm_error,float(np.max(np.abs(np.linalg.norm(chunk,axis=1)-1))))
        if norm_error>.002:raise ValueError('Unexpected final vector norms')
        audit=dict(status='PASS label-free semantic cache',rows=len(values),sha256=manifest['sha256'],max_l2_norm_error=norm_error,
            copied_rows=manifest['copied_rows'],encoded_rows=manifest['encoded_rows'],manifest_sha256=digest(DEST/'manifest.json'))
        atomic_state(ROOT/'outputs/researchforge-study/data-audit/final_semantic_index.json',audit)
        state('complete label-free cache and independent finite/norm/hash audit',audit=audit)
    except Exception as error:state('failed',error=repr(error));raise
    finally:LOCK.unlink()


if __name__=='__main__':main()
