"""Resumable label-free lexical metadata/index for the separate final corpus.

Reads only ID/date/title/abstract. No authors, roles, references or judgments.
Stateless hashing matches TRAIN; dates are used only for strict historical masks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3]
COLUMNS=['corpus_id','date','title','abstract']


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def build(source,destination,batch_size=2048,expected_ids=None,after_chunk=None):
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    config=dict(source_sha256=digest(source),batch_size=batch_size,columns=COLUMNS,dimensions=2**18,
                builder_sha256=digest(__file__),kind='Stateless nonnegative hashed unigram TF cosine; no IDF')
    cfg_path=destination/'config.json';done=destination/'manifest.json'
    if cfg_path.exists() and read(cfg_path)!=config:raise ValueError('Lexical cache configuration changed')
    if done.exists():
        report=read(done)
        if report['config']!=config:raise ValueError('Completed lexical configuration differs')
        for name,item in report['files'].items():
            if digest(destination/name)!=item['sha256']:raise ValueError('Completed lexical file changed')
        if expected_ids is not None and read(destination/'ids.json')!=expected_ids:raise ValueError('Lexical and semantic row orders differ')
        return report
    lock=destination/'build.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid()),stream)
    try:
        atomic_state(cfg_path,config);chunks=destination/'chunks';chunks.mkdir(exist_ok=True)
        vectorizer=HashingVectorizer(n_features=2**18,alternate_sign=False,norm='l2',stop_words='english',dtype=np.float32)
        paths=[];total_rows=0
        for number,batch in enumerate(pq.ParquetFile(source).iter_batches(batch_size=batch_size,columns=COLUMNS)):
            path=chunks/f'{number:06d}.npz';receipt=path.with_suffix('.json');paths.append(path)
            if receipt.exists():
                item=read(receipt)
                if digest(path)!=item['sha256'] or item['rows']!=len(batch):raise ValueError('Committed lexical chunk changed')
            else:
                rows=batch.to_pylist();texts=[(r['title'] or '')+' '+(r['abstract'] or '') for r in rows]
                matrix=vectorizer.transform(texts).tocsr();matrix.sort_indices()
                ids=np.asarray([str(r['corpus_id']) for r in rows]);dates=np.asarray([
                    int(np.datetime64(str(r['date'])[:10],'D').astype(int)) if r['date'] else 10**8 for r in rows],dtype=np.int32)
                temporary=path.with_suffix('.part')
                with temporary.open('wb') as stream:np.savez_compressed(stream,data=matrix.data,indices=matrix.indices.astype(np.int32),
                    indptr=matrix.indptr.astype(np.int64),ids=ids,dates=dates)
                temporary.replace(path)
                item=dict(rows=len(rows),missing_text=sum(not text.strip() for text in texts),sha256=digest(path))
                atomic_state(receipt,item)
            total_rows+=item['rows']
            atomic_state(destination/'progress.json',dict(stage='lexical chunks',rows=total_rows,worker_pid=os.getpid()))
            if after_chunk is not None:after_chunk(number)
        if {p.name for p in chunks.glob('*.npz')}!={p.name for p in paths}:raise ValueError('Unexpected chunk inventory')
        ids=[];dates=[];indptr=[0];missing=0
        with (destination/'data.bin').open('wb') as data,(destination/'indices.bin').open('wb') as indices:
            for path in paths:
                with np.load(path,allow_pickle=False) as saved:
                    saved['data'].tofile(data);saved['indices'].tofile(indices)
                    ids.extend(saved['ids'].tolist());dates.extend(saved['dates'].tolist())
                    indptr.extend((saved['indptr'][1:]+indptr[-1]).tolist())
                missing+=read(path.with_suffix('.json'))['missing_text']
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate corpus IDs')
        if expected_ids is not None and ids!=expected_ids:raise ValueError('Lexical and semantic row orders differ')
        np.save(destination/'dates.npy',np.asarray(dates,dtype=np.int32))
        np.save(destination/'indptr.npy',np.asarray(indptr,dtype=np.int64));atomic_state(destination/'ids.json',ids)
        names=['data.bin','indices.bin','dates.npy','indptr.npy','ids.json']
        report=dict(status='complete',config=config,rows=len(ids),nnz=indptr[-1],dimensions=2**18,missing_text=missing,
                    source_sha256=config['source_sha256'],kind=config['kind'],dtype='float32',
                    files={name:dict(bytes=(destination/name).stat().st_size,sha256=digest(destination/name)) for name in names})
        atomic_state(done,report);return report
    finally:lock.unlink()


if __name__=='__main__':
    import psutil
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'work/data/prescience/test.parquet')
    parser.add_argument('--output',type=Path,default=ROOT/'work/indices/prescience-test-hash18');args=parser.parse_args()
    if psutil.virtual_memory().available<2500*1024**2 or shutil.disk_usage(ROOT).free<2*1024**3:
        raise RuntimeError('Need2500MiB availableRAM and2GiB free disk for lexical preprocessing')
    ids=read(ROOT/'work/indices/prescience-test-text-plan/test_ids.json')
    print(json.dumps(build(args.source,args.output,expected_ids=ids),indent=2))
