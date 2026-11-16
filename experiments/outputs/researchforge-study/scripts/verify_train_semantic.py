"""Independent completed TRAIN vector byte/finite/norm/row inventory audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from supervisor_state import atomic_state
ROOT=Path(__file__).resolve().parents[3]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if __name__=='__main__':
    folder=ROOT/'work/indices/prescience-minilm';meta=ROOT/'work/indices/prescience-hash18'
    m=json.loads((folder/'manifest.json').read_text());lex=json.loads((meta/'manifest.json').read_text());ids=json.loads((meta/'ids.json').read_text())
    if m['status']!='complete' or m['completed_rows']!=m['total_rows'] or len(ids)!=m['total_rows'] or len(set(ids))!=len(ids):raise ValueError('Invalid row inventory')
    if m['config']['source_sha256']!=lex['source_sha256'] or m['config']['source_sha256']!=sha(ROOT/'work/data/prescience/train.parquet'):raise ValueError('TRAIN source checksum differs')
    path=folder/'embeddings.f16'
    if path.stat().st_size!=len(ids)*384*2 or sha(path)!=m['sha256']:raise ValueError('Vector bytes differ')
    x=np.memmap(path,mode='r',dtype=np.float16,shape=(len(ids),384));error=0.
    for start in range(0,len(x),4096):
        batch=x[start:start+4096].astype(np.float32)
        if not np.isfinite(batch).all():raise ValueError('Nonfinite vector')
        error=max(error,float(np.abs(np.linalg.norm(batch,axis=1)-1).max()))
    if error>.002:raise ValueError('Unexpected vector normalization')
    report=dict(status='PASS full TRAIN vector source/hash/finite/norm/row inventory',rows=len(x),sha256=m['sha256'],bytes=m['bytes'],
        max_l2_norm_error=error,source_sha256=m['config']['source_sha256'],manifest_sha256=sha(folder/'manifest.json'),ids_sha256=sha(meta/'ids.json'))
    atomic_state(ROOT/'outputs/researchforge-study/data-audit/train_semantic_index.json',report);print(json.dumps(report))
