"""Recover exact hashed TF counts from the existing normalized corpus index.

Re-tokenizes collection text to obtain row norms/lengths and verifies EVERY
stored posting against integer counts. Adds only two small per-document arrays.
No query relevance or held-out labels are read.
"""
import gzip
import hashlib
import json
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from exharmony_index import INDEX,DATA,AUDIT

def main():
    dest=INDEX/'bm25_stats.npz';manifest=INDEX/'bm25_stats.json'
    if dest.exists() and manifest.exists():print('Completed statistics already exist',flush=True);return
    began=time.perf_counter();meta=json.loads((INDEX/'manifest.json').read_text())
    ids=json.loads((INDEX/'ids.json').read_text());excluded=set(json.loads((AUDIT/'exharmony_excluded_corpus_ids.json').read_text()))
    original=sparse.csr_matrix((np.memmap(INDEX/'data.bin',dtype=np.float32,mode='r'),
        np.memmap(INDEX/'indices.bin',dtype=np.int32,mode='r'),np.load(INDEX/'indptr.npy',mmap_mode='r')),
        shape=(meta['rows'],meta['dimensions']),copy=False)
    vectorizer=HashingVectorizer(n_features=2**18,alternate_sign=False,norm=None,stop_words='english',dtype=np.float32)
    norms=np.empty(len(ids),dtype=np.float32);lengths=np.empty(len(ids),dtype=np.float32)
    texts=[];seen=0
    def flush():
        nonlocal seen
        if not texts:return
        x=vectorizer.transform(texts).tocsr();x.sort_indices();stop=seen+len(texts)
        norm=np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        old=original[seen:stop]
        assert np.array_equal(old.indptr,x.indptr) and np.array_equal(old.indices,x.indices)
        reconstructed=np.rint(old.data*np.repeat(norm,np.diff(x.indptr)))
        assert np.array_equal(reconstructed,x.data),'Normalized count reconstruction mismatch'
        norms[seen:stop]=norm;lengths[seen:stop]=np.asarray(x.sum(axis=1)).ravel()
        seen=stop;texts.clear()
        if seen%102400==0:print(f'Verified {seen} documents',flush=True)
    with gzip.open(DATA/'papers_collection.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['id'] in excluded:continue
            assert row['id']==ids[seen+len(texts)],'Corpus row alignment changed'
            texts.append((row.get('title') or '')+' '+(row.get('abstract') or ''))
            if len(texts)==2048:flush()
    flush();assert seen==len(ids)
    np.savez_compressed(dest,norms=norms,lengths=lengths)
    result=dict(rows=seen,verified_postings=meta['nnz'],average_document_length=float(lengths.mean()),
        zero_token_documents=int((lengths==0).sum()),seconds=time.perf_counter()-began,
        index_manifest_sha256=hashlib.sha256((INDEX/'manifest.json').read_bytes()).hexdigest(),
        stats_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        definition='Nonnegative262144-bin hashed unigram counts with English stop words removed; collisions retained')
    manifest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
