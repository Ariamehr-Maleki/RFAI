"""Build compact paper-author incidence matrices after filtered text indexing."""
import array
import gzip
import hashlib
import json
import numpy as np
from scipy import sparse
from exharmony_index import INDEX, DATA

def main():
    assert (INDEX/'manifest.json').exists(),'Text index must complete first'
    if (INDEX/'authors.manifest.json').exists():return
    ids=json.loads((INDEX/'ids.json').read_text())
    mapping={pid:i for i,pid in enumerate(ids)}
    p_rows,a_rows,author_ids=array.array('I'),array.array('I'),array.array('Q')
    with gzip.open(DATA/'authors_works_collection_ids.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);a=len(author_ids)
            author_ids.append(int(r['id'].rsplit('A',1)[-1]))
            for pid in set(r['works']):
                if pid in mapping:
                    p_rows.append(mapping[pid]);a_rows.append(a)
    m=sparse.coo_matrix((np.ones(len(p_rows),dtype=np.float32),
        (np.frombuffer(p_rows,dtype=np.uint32),np.frombuffer(a_rows,dtype=np.uint32))),
        shape=(len(ids),len(author_ids))).tocsr()
    m.sum_duplicates();assert np.all(m.data==1)
    sparse.save_npz(INDEX/'paper_authors.npz',m)
    sparse.save_npz(INDEX/'author_papers.npz',m.T.tocsr())
    np.save(INDEX/'author_ids.npy',np.frombuffer(author_ids,dtype=np.uint64))
    record=dict(papers=len(ids),authors=len(author_ids),links=m.nnz,
        authors_without_eligible_history=int(np.sum(np.asarray(m.sum(axis=0)).ravel()==0)),files={})
    for name in ['paper_authors.npz','author_papers.npz','author_ids.npy']:
        p=INDEX/name
        with p.open('rb') as f:record['files'][name]=dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
    (INDEX/'authors.manifest.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
