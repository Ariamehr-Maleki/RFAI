"""Exact, blockwise CPU semantic retrieval over historical TRAIN papers only.

Consumes a *completed* semantic_encode.py artifact. No ANN approximation and no
future rows in the seed profile; same-day/future retrieval scores are zeroed.
The caller must apply its existing strictly-positive retrieval threshold.
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]

def historical_scores(matrix, dates, rows, cutoff, block_size=4096):
    eligible=[int(i) for i in rows if dates[i] < cutoff]
    scores=np.zeros(len(dates),dtype=np.float32)
    if not eligible:return scores
    q=np.asarray(matrix[eligible],dtype=np.float32).mean(axis=0)
    q/=max(float(np.linalg.norm(q)),1e-12)
    for start in range(0,len(dates),block_size):
        stop=min(start+block_size,len(dates))
        x=np.asarray(matrix[start:stop],dtype=np.float32)
        # Re-normalize after compact float16 storage to retain cosine semantics.
        norms=np.linalg.norm(x,axis=1)
        scores[start:stop]=(x@q)/np.maximum(norms,1e-12)
    scores[dates>=cutoff]=0
    return scores

class SemanticIndex:
    def __init__(self,path=None,metadata_path=None):
        path=Path(path or ROOT/'work/indices/prescience-minilm')
        metadata_path=Path(metadata_path or ROOT/'work/indices/prescience-hash18')
        manifest_path=path/'manifest.json'
        if not manifest_path.exists():
            raise RuntimeError('Semantic embeddings are incomplete; do not use a partial corpus')
        self.manifest=json.loads(manifest_path.read_text())
        assert self.manifest['status']=='complete'
        metadata=json.loads((metadata_path/'manifest.json').read_text())
        assert self.manifest['config']['source_sha256']==metadata['source_sha256']
        self.ids=json.loads((metadata_path/'ids.json').read_text())
        self.row={pid:i for i,pid in enumerate(self.ids)}
        self.dates=np.load(metadata_path/'dates.npy',mmap_mode='r')
        assert len(self.ids)==self.manifest['total_rows']==metadata['rows']
        data=path/'embeddings.f16'
        dims=self.manifest['config']['dimensions']
        assert data.stat().st_size==len(self.ids)*dims*2==self.manifest['bytes']
        self.matrix=np.memmap(data,dtype=np.float16,mode='r',shape=(len(self.ids),dims))

    def similarities(self,historical_ids,cutoff):
        rows=[self.row[p] for p in historical_ids if p in self.row]
        return historical_scores(self.matrix,self.dates,rows,cutoff)
