"""Label-free TEST text cache: reuse exact TRAIN vectors and encode missing rows.

Never loads authors, dates, references or relevance labels. This artifact alone
cannot evaluate a query. A future scoring runner must enforce the final lock and
strict historical masks. Does not change or interrupt the active TRAIN encoder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import psutil
import numpy as np
import pyarrow.parquet as pq
from prescience_text_reuse_plan import COLUMNS,text_hash
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3]

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read_json(path):return json.loads(Path(path).read_text())

def atomic_json(path,value):
    atomic_state(path,value)

def require_train_complete(directory):
    path=directory/'manifest.json'
    if not path.exists():raise RuntimeError('TRAIN cache incomplete: wait for its final manifest')
    manifest=read_json(path)
    if manifest.get('status')!='complete' or manifest['completed_rows']!=manifest['total_rows']:
        raise RuntimeError('TRAIN cache incomplete')
    return manifest

def build(source,plan_dir,train_dir,model_dir,dest,encoder_factory,reserve_bytes=1024**3):
    train=require_train_complete(train_dir)
    plan=read_json(plan_dir/'manifest.json');model=read_json(model_dir/'download_manifest.json')
    cfg=train['config'];dims=cfg['dimensions'];batch_size=cfg['batch_size']
    if dims!=384 or cfg['stored_dtype']!='float16' or cfg['max_length']!=256 or cfg['pooling']!='attention-mask mean + L2 normalization':
        raise ValueError('Unsupported encoder configuration')
    if model['revision']!=cfg['revision'] or model['repository']!=cfg['repository']:
        raise ValueError('Model revision differs from TRAIN')
    for name,item in model['files'].items():
        if digest(model_dir/name)!=item['sha256']:raise ValueError('Model file changed: '+name)
    if cfg['source_sha256']!=plan['source_sha256']['train']:raise ValueError('TRAIN source mismatch')
    if digest(source)!=plan['source_sha256']['test']:raise ValueError('TEST source changed')
    for name,item in plan['plan_files'].items():
        if digest(plan_dir/name)!=item['sha256']:raise ValueError('Reuse plan changed: '+name)
    data=train_dir/'embeddings.f16'
    if data.stat().st_size!=train['total_rows']*dims*2 or digest(data)!=train['sha256']:
        raise ValueError('TRAIN vector file changed')
    mapping=np.load(plan_dir/'train_row_for_test_row.npy',mmap_mode='r')
    hashes=np.load(plan_dir/'test_text_sha256.npy',mmap_mode='r')
    ids=read_json(plan_dir/'test_ids.json');total=len(ids)
    if total!=plan['test_rows'] or len(mapping)!=total or len(hashes)!=total or plan['train_rows']!=train['total_rows']:
        raise ValueError('Reuse plan dimensions differ')
    if np.any(mapping < -1) or np.any(mapping>=train['total_rows']):raise ValueError('Invalid reuse row')
    matrix=np.memmap(data,dtype=np.float16,mode='r',shape=(train['total_rows'],dims))
    config=dict(cfg,source_sha256=plan['source_sha256']['test'])
    provenance=dict(config=config,train_manifest_sha256=digest(train_dir/'manifest.json'),
        plan_manifest_sha256=digest(plan_dir/'manifest.json'),
        encoder_source_sha256=digest(Path(__file__).with_name('semantic_encode.py')),
        builder_source_sha256=digest(__file__))
    provenance['atomic_writer_source_sha256']=digest(Path(__file__).with_name('supervisor_state.py'))
    dest.mkdir(parents=True,exist_ok=True)
    output=dest/'embeddings.f16';progress=dest/'progress.json';manifest_path=dest/'manifest.json'
    if manifest_path.exists():
        done=read_json(manifest_path)
        if done['provenance']!=provenance or digest(output)!=done['sha256']:
            raise ValueError('Completed TEST artifact changed')
        return done
    lock=dest/'build.lock'
    # O_EXCL prevents simultaneous writers; a stale lock needs explicit inspection.
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),purpose='label-free cache build'),f)
    try:
        completed=0;copied=0;encoded=0;hasher=hashlib.sha256()
        if progress.exists():
            previous=read_json(progress)
            state=previous
            if previous['provenance']!=provenance:raise ValueError('Resume configuration changed')
            completed=previous['completed_rows'];copied=previous['copied_rows'];encoded=previous['encoded_rows']
            if not 0<=completed<=total or output.stat().st_size<completed*dims*2:raise ValueError('Invalid checkpoint length')
            with output.open('rb') as f:
                left=completed*dims*2
                while left:
                    chunk=f.read(min(left,1024**2));hasher.update(chunk);left-=len(chunk)
            if hasher.hexdigest()!=previous['prefix_sha256']:raise ValueError('Checkpoint prefix corrupted')
        elif output.exists() and output.stat().st_size:
            raise ValueError('Untracked TEST output exists; inspect instead of overwriting')
        if shutil.disk_usage(dest).free-(total-completed)*dims*2<reserve_bytes:
            raise OSError('Insufficient disk reserve')
        encoder=None;seen=0;began=time.perf_counter()
        with output.open('r+b' if output.exists() else 'w+b') as out:
            out.truncate(completed*dims*2);out.seek(0,2)
            for batch in pq.ParquetFile(source).iter_batches(batch_size=512,columns=COLUMNS):
                stop=seen+len(batch)
                if stop<=completed:seen=stop;continue
                start=max(seen,completed);rows=batch.to_pylist()[start-seen:]
                vectors=np.empty((len(rows),dims),dtype=np.float16)
                missing=[];texts=[];copied_now=0
                for i,row in enumerate(rows):
                    j=start+i
                    if str(row['corpus_id'])!=ids[j] or text_hash(row)!=hashes[j].decode():
                        raise ValueError('Text/order differs from reuse plan')
                    prior=int(mapping[j])
                    if prior>=0:vectors[i]=matrix[prior];copied_now+=1
                    else:missing.append(i);texts.append((row.get('title') or '')+' '+(row.get('abstract') or ''))
                if missing and encoder is None:encoder=encoder_factory(cfg['threads'])
                for i in range(0,len(missing),batch_size):
                    vectors[missing[i:i+batch_size]]=encoder.encode(texts[i:i+batch_size]).astype(np.float16)
                if not np.isfinite(vectors).all():raise ValueError('Non-finite vectors')
                payload=vectors.tobytes();out.write(payload);out.flush();os.fsync(out.fileno());hasher.update(payload)
                completed=stop;copied+=copied_now;encoded+=len(missing)
                state=dict(provenance=provenance,completed_rows=completed,total_rows=total,
                    copied_rows=copied,encoded_rows=encoded,prefix_sha256=hasher.hexdigest(),
                    worker_pid=os.getpid(),session_seconds=time.perf_counter()-began)
                atomic_json(progress,state);print(json.dumps(state),flush=True);seen=stop
                if shutil.disk_usage(dest).free<reserve_bytes:raise OSError('Disk reserve reached; checkpoint saved')
                if psutil.virtual_memory().available<640*1024**2:raise MemoryError('RAM reserve reached; committed checkpoint saved')
        if completed!=total:raise ValueError('Corpus row count differs')
        state.update(status='complete',config=config,sha256=hasher.hexdigest(),bytes=output.stat().st_size,
            row_order='same as test.parquet and prescience-test-text-plan/test_ids.json',
            evaluation_status='Label-free preprocessing only; no test scores or labels accessed')
        atomic_json(manifest_path,state);return state
    finally:lock.unlink()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check-ready',action='store_true');args=ap.parse_args()
    train_dir=ROOT/'work/indices/prescience-minilm'
    require_train_complete(train_dir)
    if args.check_ready:print('TRAIN manifest complete; full build performs hash/resource verification');return
    import psutil
    if psutil.virtual_memory().available<2500*1024**2:raise RuntimeError('Wait for at least2500MiB available RAM')
    from semantic_encode import Encoder
    build(ROOT/'work/data/prescience/test.parquet',ROOT/'work/indices/prescience-test-text-plan',
          train_dir,ROOT/'work/models/all-MiniLM-L6-v2',ROOT/'work/indices/prescience-test-minilm',Encoder)

if __name__=='__main__':main()
