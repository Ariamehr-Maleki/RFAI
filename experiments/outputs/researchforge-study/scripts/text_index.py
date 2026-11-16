"""Memory-mapped stateless lexical index; never fits vocabulary/IDF on future text."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / 'work/indices/prescience-hash18'
SOURCE = ROOT / 'work/data/prescience/train.parquet'

def build():
    INDEX.mkdir(parents=True, exist_ok=True)
    if (INDEX/'manifest.json').exists():
        return
    vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False,
                                  norm='l2', stop_words='english', dtype=np.float32)
    ids, dates, indptr = [], [], [0]
    missing_text = 0
    with (INDEX/'data.bin').open('wb') as values, (INDEX/'indices.bin').open('wb') as indices:
        for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=2048, columns=['corpus_id','date','title','abstract']):
            rows = batch.to_pylist()
            texts = [(r['title'] or '')+' '+(r['abstract'] or '') for r in rows]
            x = vectorizer.transform(texts).tocsr()
            x.sort_indices()
            x.data.tofile(values)
            x.indices.astype(np.int32).tofile(indices)
            indptr.extend((x.indptr[1:]+indptr[-1]).tolist())
            ids.extend(str(r['corpus_id']) for r in rows)
            dates.extend(int(np.datetime64(str(r['date'])[:10], 'D').astype(int)) if r['date'] else 10**8 for r in rows)
            missing_text += sum(not t.strip() for t in texts)
            if len(ids) % 20480 == 0:
                print(f'Indexed {len(ids)} papers', flush=True)
    np.save(INDEX/'indptr.npy', np.array(indptr,dtype=np.int64))
    np.save(INDEX/'dates.npy', np.array(dates,dtype=np.int32))
    (INDEX/'ids.json').write_text(json.dumps(ids))
    source_manifest=json.loads(SOURCE.with_suffix('.parquet.manifest.json').read_text())
    record=dict(rows=len(ids),nnz=indptr[-1],dimensions=2**18,missing_text=missing_text,
                source_sha256=source_manifest['sha256'],kind='Stateless nonnegative hashed unigram TF cosine; no IDF',
                dtype='float32',files={})
    for p in INDEX.iterdir():
        if p.is_file():
            with p.open('rb') as f:
                record['files'][p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
    (INDEX/'manifest.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)

class TextIndex:
    def __init__(self, path=INDEX):
        manifest=json.loads((path/'manifest.json').read_text())
        self.ids=json.loads((path/'ids.json').read_text())
        self.row={pid:i for i,pid in enumerate(self.ids)}
        self.dates=np.load(path/'dates.npy',mmap_mode='r')
        self.matrix=sparse.csr_matrix((np.memmap(path/'data.bin',dtype=np.float32,mode='r'),
            np.memmap(path/'indices.bin',dtype=np.int32,mode='r'),np.load(path/'indptr.npy',mmap_mode='r')),
            shape=(manifest['rows'],manifest['dimensions']),copy=False)

    def similarities(self, historical_ids, cutoff):
        rows=[self.row[p] for p in historical_ids if p in self.row and self.dates[self.row[p]] < cutoff]
        if not rows:
            return np.zeros(len(self.ids),dtype=np.float32)
        q=np.asarray(self.matrix[rows].mean(axis=0)).reshape(-1)
        q/=max(float(np.linalg.norm(q)),1e-12)
        scores=self.matrix@q
        scores[self.dates>=cutoff]=0
        return scores

if __name__=='__main__':
    build()
