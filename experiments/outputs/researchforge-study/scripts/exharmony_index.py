"""Stream the full exHarmony collection to a disk-backed lexical index.

Excludes IDs and normalized-title matches to all queries using the frozen audit.
No qrels or model evaluation. Matching abstracts are audited, not used as labels.
"""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from exharmony_prepare import ROOT, DATA, OUT as AUDIT, norm

INDEX=ROOT/'work/indices/exharmony-hash18'

def main():
    INDEX.mkdir(parents=True,exist_ok=True)
    if (INDEX/'manifest.json').exists():
        print('Completed index already present; no rebuild',flush=True);return
    excluded=set(json.loads((AUDIT/'exharmony_excluded_corpus_ids.json').read_text()))
    query_abstracts={}
    with (DATA/'papers_test.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            key=norm(r.get('title'))
            query_abstracts.setdefault(key,set()).add(hashlib.sha256(norm(r.get('abstract')).encode()).hexdigest())
    vectorizer=HashingVectorizer(n_features=2**18,alternate_sign=False,norm='l2',stop_words='english',dtype=np.float32)
    ids,indptr,texts=[],[0],[]
    overlap_exact=0;examples=[]
    with (INDEX/'data.bin').open('wb') as values,(INDEX/'indices.bin').open('wb') as indices:
        def flush():
            if not texts:return
            x=vectorizer.transform(texts).tocsr();x.sort_indices()
            x.data.tofile(values);x.indices.astype(np.int32).tofile(indices)
            indptr.extend((x.indptr[1:]+indptr[-1]).tolist());texts.clear()
        with gzip.open(DATA/'papers_collection.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['id'] in excluded:
                    exact=hashlib.sha256(norm(r.get('abstract')).encode()).hexdigest() in query_abstracts.get(norm(r.get('title')),set())
                    overlap_exact+=exact
                    if len(examples)<10:examples.append(dict(id=r['id'],title=r['title'],same_normalized_abstract=exact))
                    continue
                ids.append(r['id']);texts.append((r.get('title') or '')+' '+(r.get('abstract') or ''))
                if len(texts)>=2048:
                    flush()
                    if len(ids)%20480==0:print(f'Indexed {len(ids)} papers',flush=True)
        flush()
    np.save(INDEX/'indptr.npy',np.array(indptr,dtype=np.int64))
    # No dates released; zero denotes the fixed collection, never inferred dates.
    np.save(INDEX/'dates.npy',np.zeros(len(ids),dtype=np.int32))
    (INDEX/'ids.json').write_text(json.dumps(ids))
    report=dict(rows=len(ids),dimensions=2**18,nnz=indptr[-1],excluded_documents=len(excluded),
        exact_normalized_title_and_abstract_overlap=overlap_exact,overlap_examples=examples,
        note='Title exclusion is conservative: not every repeated title proves duplicate content.',
        kind='Stateless nonnegative hashed unigram TF cosine; no IDF',files={})
    for p in INDEX.iterdir():
        if p.is_file():
            with p.open('rb') as f:report['files'][p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
    (INDEX/'manifest.json').write_text(json.dumps(report,indent=2))
    (AUDIT/'exharmony_text_overlap.json').write_text(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
