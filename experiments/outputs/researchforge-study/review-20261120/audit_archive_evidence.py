"""Read-only, streaming archive audit and independently coded ranking metrics.

Does not import the original evaluation implementation, fit a model, extract
provider data, or touch an unevaluated cohort. Reads one prediction file at a time.
"""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import argparse, hashlib, json, math, io, time, re
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ARCHIVES=['researchforge-reproducibility.zip','ResearchForge_Review_Extension_20261108.zip']

def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')

def digest_file(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def inventory():
    combined={};overview=[]
    for archive in ARCHIVES:
        path=STUDY/'deliverables'/archive
        with ZipFile(path) as z:
            names=z.namelist();assert len(names)==len(set(names))
            manifest='PACKAGE_MANIFEST.json' if archive==ARCHIVES[0] else 'EXTENSION_MANIFEST.json'
            expected=json.loads(z.read(manifest))['files']
            assert set(names)==set(expected)|{manifest}
            records=[]
            for i,n in enumerate(names):
                p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
                h=hashlib.sha256();count=0
                with z.open(n) as f:
                    while chunk:=f.read(1024**2):h.update(chunk);count+=len(chunk)
                actual=h.hexdigest()
                if n!=manifest:assert actual==expected[n]['sha256'] and count==expected[n]['bytes'],n
                records.append({'archive':archive,'source_file':n,'sha256':actual,'bytes':count})
                if i%1000==0:print('verified',archive,i,flush=True)
            combined[archive]=records
            overview.append({'archive':archive,'sha256':digest_file(path),'compressed_bytes':path.stat().st_size,
                             'expanded_bytes':sum(r['bytes'] for r in records),'files':len(records),'all_manifest_payloads_verified':True})
        write('archive-integrity-progress.json',overview)
    write('joint-file-manifest.json',combined)
    write('archive-integrity.json',overview)
    return combined

def metrics(row,expert=False,candidate_recall=None):
    truth=set(row['truth']);pred=row['predicted'];assert len(pred)==len(set(pred))
    assert truth
    hits=[i+1 for i,a in enumerate(pred) if a in truth]
    n=len(truth);out={}
    ks=[10,20,100] if expert else [10,1000]
    for k in ks:
        gain=math.fsum(1/math.log2(i+1) for i in hits if i<=k)
        ideal=math.fsum(1/math.log2(i+1) for i in range(1,min(k,n)+1))
        out['ndcg'+str(k)]=gain/ideal
        if expert:
            out['map'+str(k)]=math.fsum(j/i for j,i in enumerate(hits,1) if i<=k)/n
            out['recall'+str(k)]=sum(i<=k for i in hits)/n
            out['rr'+str(k)]=1/hits[0] if hits and hits[0]<=k else 0.
    if not expert:
        out.update(rprecision=sum(i<=n for i in hits)/n,recall10=sum(i<=10 for i in hits)/n,
                   recall50=sum(i<=50 for i in hits)/n)
        new=set(row.get('new_truth',[]))
        if new:out['new_recall10']=len(new&set(pred[:10]))/len(new)
    # exHarmony predictions retain top 100, whereas the pool has 1,000 authors.
    # Its candidate recall MUST come from the query cache, joined by query ID.
    if expert:
        if candidate_recall is None:raise ValueError('Full query-pool recall required for exHarmony')
        out['candidate_recall']=candidate_recall
    else:out['candidate_recall']=len(truth&set(pred))/n
    return out

def audit_predictions():
    rows=[];errors=[]
    with ZipFile(STUDY/'deliverables'/ARCHIVES[0]) as z:
        for dataset in ['prescience','exHarmony','ogbl-collab']:
            base=f'outputs/researchforge-study/results/final/{dataset}/'
            report=json.loads(z.read(base+'report.json'))
            candidate_recalls={}
            if dataset=='exHarmony':
                caches=[n for n in z.namelist() if n.startswith(base+'queries/') and n.endswith('.json') and not n.endswith('.receipt.json')]
                for n in caches:
                    q=json.loads(z.read(n));assert q['qid'] not in candidate_recalls
                    pool=set(q['candidates']);assert len(pool)==1000
                    candidate_recalls[q['qid']]={j:len(pool&set(t))/len(set(t)) for j,t in q['truth'].items() if t}
                assert len(candidate_recalls)==1589
            names=[n for n in z.namelist() if n.startswith(base) and n.endswith(('.predictions.json','.predictions.npz'))]
            for i,n in enumerate(names):
                filename=n[len(base):]
                match=re.fullmatch(r'(.+)\.s(42|43|44)\.predictions\.(json|npz)',filename)
                if not match:raise ValueError('Unexpected prediction filename: '+n)
                stem,seed,_=match.groups()
                if dataset=='ogbl-collab':
                    with np.load(io.BytesIO(z.read(n)),allow_pickle=False) as a:
                        threshold=np.sort(a['negative'])[-50]
                        result={'hits50':float(np.mean(a['positive']>threshold))};count=len(a['positive'])
                    expected={'hits50':report['per_seed'][stem][seed]};max_row_error=None
                else:
                    predictions=json.loads(z.read(n));assert len({p['qid'] for p in predictions})==len(predictions)
                    values={};max_row_error=0.
                    for row in predictions:
                        pool_recall=candidate_recalls[row['qid']][stem.split('.',1)[0]] if dataset=='exHarmony' else None
                        for k,v in metrics(row,dataset=='exHarmony',pool_recall).items():
                            values.setdefault(k,[]).append(v)
                            if k in row and row[k] is not None:max_row_error=max(max_row_error,abs(v-row[k]))
                    result={k:math.fsum(v)/len(v) for k,v in values.items()};count=len(predictions)
                    if dataset=='exHarmony':
                        judgment,method=stem.split('.',1);expected=report['per_seed'][judgment][method][seed]
                    else:expected=report['per_seed'][stem][seed]
                    del predictions
                diffs={k:abs(v-expected[k]) for k,v in result.items() if expected.get(k) is not None}
                max_error=max(diffs.values())
                record={'archive':ARCHIVES[0],'source_file':n,'dataset':dataset,'method':stem,'seed':seed,
                        'queries_or_positive_edges':count,'recomputed':result,'max_summary_error':max_error,'max_row_error':max_row_error,
                        'candidate_recall_source':'Shared full 1000-candidate query JSON joined by qid' if dataset=='exHarmony' else 'Complete saved candidate ranking'}
                rows.append(record)
                if max_error>1e-12 or (max_row_error is not None and max_row_error>1e-12):errors.append(record)
                if i%10==0:print('metrics',dataset,i+1,'/',len(names),flush=True)
            write('independent-metrics-progress.json',{'files':len(rows),'failures':errors})
    with ZipFile(STUDY/'deliverables'/ARCHIVES[1]) as z:
        base='outputs/researchforge-study/review-20261107/'
        report=json.loads(z.read(base+'official-comparator-results.json'))
        for track,summary in report['results'].items():
            n=base+track+'.predictions.json';predictions=json.loads(z.read(n));values={}
            for row in predictions:
                for k,v in metrics(row).items():values.setdefault(k,[]).append(v)
            actual={k:math.fsum(v)/len(v) for k,v in values.items()}
            error=max(abs(v-summary['metrics'][k]) for k,v in actual.items())
            record={'archive':ARCHIVES[1],'source_file':n,'dataset':'prescience-official','method':track,
                    'queries_or_positive_edges':len(predictions),'recomputed':actual,'max_summary_error':error}
            rows.append(record)
            if error>1e-12:errors.append(record)
    write('independent-metrics.json',{'status':'PASS' if not errors else 'FAIL','files':len(rows),'results':rows,'failures':errors,
        'scope':'Independent metric recomputation from archived rankings/scores, not raw-data feature reconstruction or model execution.',
        'script_sha256':digest_file(Path(__file__))})
    print('metric audit finished',len(rows),'files',len(errors),'failures',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['integrity','metrics','all'],default='all');args=p.parse_args()
    if args.mode in ['all','integrity']:inventory()
    if args.mode in ['all','metrics']:audit_predictions()
