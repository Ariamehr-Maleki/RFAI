"""Descriptive audit of a completed DEVELOPMENT cohort, no final labels.

Reads committed metadata one query at a time. Training candidate recall is
recorded before positive injection; validation recall is checked against IDs.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'outputs/researchforge-study'
BASE=STUDY/'results/prescience-expanded-1200-1000-500-1000'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def audit(mode):
    folder=BASE/mode
    collection=read(folder/'collection_report.json')
    plan=read(BASE/'query_manifest.json')
    if collection['status']!='Development collection complete; model fitting pending':raise ValueError('Incomplete collection')
    if sha(BASE/'query_manifest.json')!=collection['query_manifest_sha256']:raise ValueError('Query manifest changed')
    result=dict(status='PASS descriptive DEVELOPMENT cohort and receipt audit; no final evaluation',mode=mode,
        source_sha256={'query_manifest.json':sha(BASE/'query_manifest.json'),'collection_report.json':sha(folder/'collection_report.json')},
        partitions={},limitations=['Training recall precedes positive injection; validation recall uses untouched retrieved candidates.',
            'Query counts and author overlap describe the temporal development sample, not final-test representativeness.',
            'No model rankings or statistical superiority assessed.'])
    authors={}
    for part,key in [('train','train_pool'),('valid','validation')]:
        queries=plan[key];authors[part]=set();truth_sizes=[];candidate_sizes=[];recalls=[];injections=[];histories=[];dates=[];new_count=0;repeat_count=0
        manifest_ids={str(q['qid']) for q in queries}
        if len(manifest_ids)!=len(queries):raise ValueError('Duplicate query IDs')
        cache=folder/(part+'_queries')
        if {p.stem for p in cache.glob('*.receipt')}!=manifest_ids:raise ValueError('Incomplete/extra committed cohort')
        receipt_hash=hashlib.sha256()
        for q in queries:
            meta=cache/(str(q['qid'])+'.json');receipt=meta.with_suffix('.receipt');r=read(receipt)
            if sha(meta)!=r['metadata_sha256'] or sha(meta.with_suffix('.npz'))!=r['features_sha256']:raise ValueError('Committed cache changed')
            receipt_hash.update((receipt.name+':'+sha(receipt)+'\n').encode())
            m=read(meta)
            if any(m[k]!=q[k] for k in ['qid','seed','date','truth']):raise ValueError('Metadata/query plan differs')
            if (m['date']<plan['split_day'])!=(part=='train'):raise ValueError('Wrong temporal partition')
            candidates=m['candidates'];truth=set(m['truth']);new=set(m['new_truth'])
            if not truth or len(candidates)!=len(set(candidates)) or not new<=truth:raise ValueError('Invalid candidate/relevance metadata')
            if part=='valid':
                if m['training_positive_injections']!=0:raise ValueError('Validation injection')
                if abs(m['candidate_recall']-len(set(candidates)&truth)/len(truth))>1e-12:raise ValueError('Recall differs')
            authors[part].add(m['seed']);dates.append(m['date']);histories.append(m['seed_history_count'])
            truth_sizes.append(len(truth));candidate_sizes.append(len(candidates));recalls.append(m['candidate_recall']);injections.append(m['training_positive_injections'])
            new_count+=bool(new);repeat_count+=bool(truth-new)
        epoch=dt.date(1970,1,1)
        result['partitions'][part]=dict(query_count=len(queries),distinct_seed_authors=len(authors[part]),
            date_start=str(epoch+dt.timedelta(days=min(dates))),date_end=str(epoch+dt.timedelta(days=max(dates))),
            truth_size_mean=statistics.mean(truth_sizes),truth_size_median=statistics.median(truth_sizes),
            candidates_min=min(candidate_sizes),candidates_max=max(candidate_sizes),
            candidate_recall_mean=statistics.mean(recalls),total_positive_injections=sum(injections),
            queries_with_injections=sum(i>0 for i in injections),queries_with_new_truth=new_count,queries_with_repeat_truth=repeat_count,
            history_query_counts=dict(Counter('1-3' if h<=3 else '4-10' if h<=10 else '11+' for h in histories)),
            receipts_sha256_in_manifest_order=receipt_hash.hexdigest())
    result['seed_author_overlap_between_temporal_partitions']=len(authors['train']&authors['valid'])
    result['audit_code_sha256']=sha(Path(__file__))
    atomic_state(STUDY/'data-audit'/f'expanded-{mode}-cohort.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['full','no_content'],required=True)
    audit(ap.parse_args().mode)
