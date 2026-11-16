"""Inventory label-free text reuse for the distinct official test companion corpus.

Reads ONLY corpus_id,title,abstract in both Parquet files. No authors, roles,
references, citation trajectories, qrels, target selection or evaluation. Shared
embeddings may be reused only when corpus ID AND exact encoder-input bytes match.
This creates a preprocessing plan; it does not freeze models or unseal labels.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'work/data/prescience'
DEST=ROOT/'work/indices/prescience-test-text-plan'
COLUMNS=['corpus_id','title','abstract']


def text_hash(row):
    text=(row.get('title') or '')+' '+(row.get('abstract') or '')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def reuse_row(row,training):
    prior=training.get(str(row['corpus_id']))
    return prior[0] if prior is not None and prior[1]==text_hash(row) else -1


def text_rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024,columns=COLUMNS):
        yield from batch.to_pylist()


def main():
    start=time.perf_counter();DEST.mkdir(parents=True,exist_ok=True)
    train={};train_ids=[]
    for i,row in enumerate(text_rows(DATA/'train.parquet')):
        pid=str(row['corpus_id'])
        if pid in train:raise ValueError('Duplicate TRAIN ID; cache mapping ambiguous')
        train[pid]=(i,text_hash(row));train_ids.append(pid)
    expected=json.loads((ROOT/'work/indices/prescience-hash18/ids.json').read_text())
    if train_ids!=expected:raise ValueError('TRAIN row order does not match active embedding/index order')
    ids=[];mapping=[];hashes=[];shared_ids=0;changed_text=0;seen=set()
    for row in text_rows(DATA/'test.parquet'):
        pid=str(row['corpus_id'])
        if pid in seen:raise ValueError('Duplicate TEST corpus ID; resolve before building index')
        seen.add(pid);ids.append(pid);sha=text_hash(row);hashes.append(sha)
        reused=reuse_row(row,train);mapping.append(reused)
        shared_ids+=pid in train;changed_text+=(pid in train and reused<0)
    mapping=np.asarray(mapping,dtype=np.int32)
    np.save(DEST/'train_row_for_test_row.npy',mapping)
    (DEST/'test_ids.json').write_text(json.dumps(ids))
    # Fixed-length byte array avoids JSON overhead; it contains hashes, not text.
    np.save(DEST/'test_text_sha256.npy',np.asarray(hashes,dtype='S64'))
    result=dict(status='Label-free text preprocessing inventory only; no test labels or scores accessed',
                columns_read=COLUMNS,train_rows=len(train),test_rows=len(ids),shared_corpus_ids=shared_ids,
                exact_text_reusable_rows=int((mapping>=0).sum()),shared_ids_with_changed_text=changed_text,
                rows_needing_encoding=int((mapping<0).sum()),additional_float16_bytes=int((mapping<0).sum())*384*2,
                full_test_float16_bytes=len(ids)*384*2,wall_seconds=time.perf_counter()-start,
                source_sha256={split:json.loads((DATA/f'{split}.parquet.manifest.json').read_text())['sha256'] for split in ['train','test']},
                plan_files={},requirements=['TRAIN embedding manifest must be complete before reuse.',
                'Pinned model revision, tokenizer, truncation, pooling and normalization must match.',
                'Future query/candidate scoring must still mask same-day/future papers and exclude target text.',
                'No supervised model choice may use held-out labels; final gate remains mandatory.',
                'Official test corpus differs from TRAIN; do not silently score test with TRAIN-only candidate coverage.'])
    for p in DEST.iterdir():
        if p.suffix=='.npy' or p.name=='test_ids.json':
            with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
            result['plan_files'][p.name]=dict(bytes=p.stat().st_size,sha256=sha)
    (DEST/'manifest.json').write_text(json.dumps(result,indent=2))
    (ROOT/'outputs/researchforge-study/data-audit/prescience_text_reuse_plan.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
