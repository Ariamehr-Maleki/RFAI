"""Development-only lexical + semantic + temporal/network ranking.

Reads TRAIN only. Checkpoints each query. Test evaluation requires a separate lock.
"""
import argparse
import bisect
import collections
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prescience_pilot import (ROOT, OUT, DATA, load_data, make_queries, sample_queries,
    day, log, rank_metrics, evaluate, cluster_difference)
from text_index import TextIndex
from semantic_index import SemanticIndex
from threadpoolctl import threadpool_limits

FEATURES=['frequency10','frequency45','frequency_all','recency_collaboration',
    'common_neighbors10','jaccard10','log_degree_product','log_seed_history',
    'log_candidate_history','log_candidate_degree','category_cosine','activity_decay',
    'history_balance','reverse_frequency10','lexical_mean10','lexical_max10',
    'lexical_retrieval','autoresearch_description','citation_pull','rrf',
    'semantic_mean10','semantic_max10','semantic_retrieval']
ASSUMPTIONS={
    'status':'Description-based reimplementation, NOT exact official Autoresearch reproduction',
    'frequency':'Unnormalised coauthor counts over last45 papers; deterministic author-id ties.',
    'propagation':'Top150 direct contributors; outgoing45-paper counts L1-normalized. Hop2=0.6*source mass*transition. Top100 hop2 contributors; hop3=0.3*hop2 mass*transition.',
    'citations':'10 unique references most recently cited in seed prior publications (citing date then referenced date then ID). Add 1 per referenced-paper author; boost weight was unspecified.',
    'activity':'Multiply summed frequency,hop2,hop3,citation scores by exp2(-days_since_last_paper/(2.5*365.25)).',
    'missing_details':'Transition normalization, citation weight, hop3 damping interpretation and tie handling are not fully specified in v2. Sensitivity and official code comparison remain required.',
    'source':'https://arxiv.org/html/2602.20459v2#A3.SS2'}

class QueryEngine:
    def __init__(self,papers,histories,dates,index,semantic):
        self.papers,self.histories,self.dates,self.index=papers,histories,dates,index
        self.semantic=semantic
        assert self.index.ids==semantic.ids,'Both text indices must have identical row order'
        self.author_ids=sorted(histories)
        self.earliest=np.array([dates[a][0] for a in self.author_ids])

    def run(self,q,n_candidates,seed,training):
        cutoff,source=q['date'],q['seed']
        papers,histories,dates,index=self.papers,self.histories,self.dates,self.index
        cache={}
        def profile(a):
            if a in cache:
                return cache[a]
            end=bisect.bisect_left(dates.get(a,[]),cutoff)
            ids=histories.get(a,[])[:end]
            f10,f45,topics=collections.Counter(),collections.Counter(),collections.Counter()
            for pid in ids[-45:]:
                f45.update(b for b in papers[pid]['authors'] if b!=a)
            for pid in ids[-10:]:
                f10.update(b for b in papers[pid]['authors'] if b!=a)
                topics.update(papers[pid]['categories'])
            cache[a]=dict(f10=f10,f45=f45,topics=topics,count=end,ids10=ids[-10:],
                          last=papers[ids[-1]]['date'] if ids else -10**6)
            return cache[a]
        sp=profile(source)
        source_ids=histories[source][:sp['count']]
        allfreq,last=collections.Counter(),{}
        cited={}
        for pid in source_ids:
            p=papers[pid]
            assert p['date']<cutoff
            for a in p['authors']:
                if a!=source:
                    allfreq[a]+=1
                    last[a]=max(last.get(a,-10**6),p['date'])
            for ref in p.get('refs',()):
                if ref in papers and papers[ref]['date']<cutoff:
                    cited[ref]=max(cited.get(ref,-10**6),p['date'])
        def ordered(scores):
            return sorted(scores,key=lambda a:(-scores[a],a))
        hop2,hop3=collections.Counter(),collections.Counter()
        for a in ordered(sp['f45'])[:150]:
            outgoing=profile(a)['f45']
            scale=.6*sp['f45'][a]/max(1,sum(outgoing.values()))
            for b,v in outgoing.items():
                hop2[b]+=scale*v
        hop2.pop(source,None)
        for a in ordered(hop2)[:100]:
            outgoing=profile(a)['f45']
            scale=.3*hop2[a]/max(1,sum(outgoing.values()))
            for b,v in outgoing.items():
                hop3[b]+=scale*v
        citation=collections.Counter()
        refs=sorted(cited,key=lambda p:(-cited[p],-papers[p]['date'],p))[:10]
        for pid in refs:
            citation.update(papers[pid]['authors'])
        auto=sp['f45']+hop2+hop3+citation
        auto.pop(source,None)
        # Last-publication dates require no expensive profile construction.
        for a in list(auto):
            end=bisect.bisect_left(dates.get(a,[]),cutoff)
            if not end:
                del auto[a]
            else:
                auto[a]*=2**(-(cutoff-dates[a][end-1])/(2.5*365.25))
        similarity=index.similarities(sp['ids10'],cutoff)
        k=min(2000,len(similarity))
        top=np.argpartition(-similarity,k-1)[:k]
        top=sorted(top,key=lambda i:(-similarity[i],index.ids[i]))
        lexical=collections.Counter()
        for i in top:
            if similarity[i]<=0:
                continue
            p=papers.get(index.ids[i])
            if p is not None and p['date']<cutoff:
                for a in p['authors']:
                    if a!=source:
                        lexical[a]+=float(similarity[i])
        semantic_similarity=self.semantic.similarities(sp['ids10'],cutoff)
        semantic_scores=collections.Counter()
        top_sem=np.argpartition(-semantic_similarity,k-1)[:k]
        top_sem=sorted(top_sem,key=lambda i:(-semantic_similarity[i],index.ids[i]))
        for i in top_sem:
            if semantic_similarity[i]<=0:continue
            p=papers.get(index.ids[i])
            if p is not None and p['date']<cutoff:
                for a in p['authors']:
                    if a!=source:semantic_scores[a]+=float(semantic_similarity[i])
        rankings=[ordered(auto),ordered(lexical),ordered(sp['f10']),ordered(semantic_scores)]
        rrf=collections.Counter()
        for ranking in rankings:
            for rank,a in enumerate(ranking):
                rrf[a]+=1/(60+rank+1)
        def eligible(a):
            return a!=source and a in dates and dates[a][0]<cutoff
        rng=np.random.default_rng(int(hashlib.sha256(f'{seed}:{q["qid"]}'.encode()).hexdigest()[:8],16))
        padding=None
        def pad(ranking):
            nonlocal padding
            result=[]
            used={source}
            for a in ranking:
                if a not in used and eligible(a):
                    result.append(a);used.add(a)
                    if len(result)==n_candidates:
                        return result
            if padding is None:
                padding=[self.author_ids[i] for i in rng.permutation(np.flatnonzero(self.earliest<cutoff))]
            for a in padding:
                if a not in used:
                    result.append(a);used.add(a)
                    if len(result)==n_candidates:
                        break
            return result
        standalone={'frequency10_standalone':pad(ordered(sp['f10'])),
                    'autoresearch_description_standalone':pad(rankings[0]),
                    'lexical_standalone':pad(rankings[1]), 'semantic_standalone':pad(rankings[3]),
                    'rrf_standalone':pad(ordered(rrf))}
        candidates=standalone['rrf_standalone'].copy()
        recall=len(set(candidates)&set(q['truth']))/len(q['truth'])
        injections=0
        if training:
            missing=sorted(set(q['truth'])-set(candidates))
            candidates.extend(missing)
            injections=len(missing)
        candidates.sort()
        sx=set(sp['f10']);st=sp['topics'];stnorm=sum(v*v for v in st.values())
        rows=[]
        for a in candidates:
            cp=profile(a);cx=set(cp['f10']);ct=cp['topics']
            inter=len(sx&cx)
            denom=math.sqrt(stnorm*sum(v*v for v in ct.values()))
            ls=[float(similarity[index.row[p]]) for p in cp['ids10'] if p in index.row]
            ss=[float(semantic_similarity[index.row[p]]) for p in cp['ids10'] if p in index.row]
            sn,cn=sp['count'],cp['count']
            rows.append([sp['f10'][a],sp['f45'][a],allfreq[a],1/(1+cutoff-last[a]) if a in last else 0,
                inter,inter/max(1,len(sx|cx)),math.log1p(len(sx)*len(cx)),math.log1p(sn),math.log1p(cn),
                math.log1p(len(cx)),sum(v*ct[k] for k,v in st.items())/denom if denom else 0,
                2**(-(cutoff-cp['last'])/(2.5*365.25)),min(sn,cn)/max(1,sn,cn),cp['f10'][source],
                float(np.mean(ls)) if ls else 0,max(ls,default=0),lexical[a],auto[a],citation[a],rrf[a],
                float(np.mean(ss)) if ss else 0,max(ss,default=0),semantic_scores[a]])
        metadata=dict(**q,candidates=candidates,candidate_recall=recall,
            new_truth=[a for a in q['truth'] if allfreq[a]==0],seed_history_count=sp['count'],
            training_positive_injections=injections,standalone=standalone)
        return np.asarray(rows,dtype=np.float32),np.array([a in q['truth'] for a in candidates],dtype=np.int8),metadata

def collect(queries,engine,run,config,training):
    folder=run/('train_queries' if training else 'valid_queries')
    folder.mkdir(exist_ok=True)
    xs,ys,groups,meta=[],[],[],[]
    for i,q in enumerate(queries):
        f=folder/(q['qid']+'.npz');j=f.with_suffix('.json')
        if f.exists() and j.exists():
            with np.load(f) as z:
                x,y=z['X'],z['y']
            m=json.loads(j.read_text())
        else:
            before=time.perf_counter()
            x,y,m=engine.run(q,config['candidates'],config['seed'],training)
            m['seconds']=time.perf_counter()-before
            with f.with_suffix('.part').open('wb') as handle:
                np.savez_compressed(handle,X=x,y=y)
            f.with_suffix('.part').replace(f)
            j.with_suffix('.part').write_text(json.dumps(m))
            j.with_suffix('.part').replace(j)
        xs.append(x);ys.append(y);groups.append(len(y));meta.append(m)
        log(f'{"train" if training else "validation"} {i+1}/{len(queries)}; {m.get("seconds",0):.2f}s/query; coverage {m["candidate_recall"]:.3f}')
    return np.concatenate(xs),np.concatenate(ys),np.array(groups),meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-queries',type=int,default=200)
    ap.add_argument('--valid-queries',type=int,default=100)
    ap.add_argument('--candidates',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args();began=time.perf_counter()
    semantic=SemanticIndex() # Reject partial embeddings before creating an experiment.
    run=OUT/'results'/f'hybrid-graph-dedup-{args.train_queries}-{args.valid_queries}-{args.candidates}-s{args.seed}'
    run.mkdir(parents=True,exist_ok=True)
    config=dict(**vars(args),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_loader_sha256=hashlib.sha256(Path(__file__).with_name('prescience_pilot.py').read_bytes()).hexdigest(),
        author_policy='Unique author IDs per paper, original first-occurrence order',
        semantic_artifact={k:semantic.manifest[k] for k in ['config','sha256','bytes','total_rows']},
        training_threads=1,numeric_threads=1,
        semantic_index_sha256=hashlib.sha256(Path(__file__).with_name('semantic_index.py').read_bytes()).hexdigest(),
        candidate_policy='RRF(auto_description,lexical,frequency10,MiniLM),k=60,top2000 positive-cosine papers per text channel',selection='validation ndcg1000')
    if (run/'config.json').exists():
        assert json.loads((run/'config.json').read_text())==config,'Code/config changed: use a fresh run directory'
    else:
        (run/'config.json').write_text(json.dumps(config,indent=2))
    papers,histories,dates=load_data()
    for batch in pq.ParquetFile(DATA/'train.parquet').iter_batches(batch_size=2048,columns=['corpus_id','key_references']):
        for p in batch.to_pylist():
            if p['corpus_id'] in papers:
                papers[p['corpus_id']]['refs']=tuple(r['corpus_id'] for r in p['key_references'] or [] if r['corpus_id'])
    index=TextIndex();engine=QueryEngine(papers,histories,dates,index,semantic)
    queries=make_queries(papers,histories,dates);split=day('2024-07-01')
    train=sample_queries([q for q in queries if q['date']<split],args.train_queries,args.seed)
    valid=sample_queries([q for q in queries if q['date']>=split],args.valid_queries,args.seed)
    tx,ty,tg,tm=collect(train,engine,run,config,True)
    vx,vy,vg,vm=collect(valid,engine,run,config,False)
    np.savez_compressed(run/'features.npz',train_X=tx,train_y=ty,train_group=tg,valid_X=vx,valid_y=vy,valid_group=vg)
    predictions={name:vx[:,idx] for name,idx in [('frequency10',0),('frequency45',1),('frequency_all',2),
        ('common_neighbors10',4),('category_cosine',10),('lexical_mean10',14),('autoresearch_description',17),('rrf',19),('semantic_mean10',20)]}
    predictions['random']=np.random.default_rng(args.seed).random(len(vx))
    tuning=[];cols=list(range(len(FEATURES)))
    # Remove aggregate scores too: otherwise the ablated evidence would leak through fusion.
    settings={'lambdamart':cols,'binary_lightgbm':cols,
              'lambdamart_without_text':[i for i in cols if i not in [14,15,16,19,20,21,22]],
              'lambdamart_without_semantic':[i for i in cols if i not in [19,20,21,22]],
              'lambdamart_without_lexical':[i for i in cols if i not in [14,15,16,19]],
              'lambdamart_without_recency':[i for i in cols if i not in [3,11,17,19]],
              'lambdamart_text_only':[14,15,16,20,21,22]}
    for name,selected in settings.items():
        best=None
        ranking=name!='binary_lightgbm'
        for leaves,trees in [(7,60),(15,100),(31,100)]:
            model=(LGBMRanker if ranking else LGBMClassifier)(n_estimators=trees,num_leaves=leaves,
                learning_rate=.05,reg_lambda=1,random_state=args.seed,n_jobs=1,verbosity=-1,
                deterministic=True,force_col_wise=True,objective='lambdarank' if ranking else 'binary')
            start=time.perf_counter()
            model.fit(tx[:,selected],ty,**({'group':tg} if ranking else {}))
            scores=model.predict(vx[:,selected]) if ranking else model.predict_proba(vx[:,selected])[:,1]
            agg,_=evaluate(scores,vg,vm)
            tuning.append(dict(method=name,leaves=leaves,trees=trees,seconds=time.perf_counter()-start,**agg))
            if best is None or agg['ndcg1000']>best[0]:
                best=agg['ndcg1000'],scores,model
        predictions[name]=best[1]
        best[2].booster_.save_model(str(run/f'{name}.txt'))
    best=None
    for c in [.1,1,10]:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=500,random_state=args.seed))
        model.fit(tx,ty);scores=model.predict_proba(vx)[:,1];agg,_=evaluate(scores,vg,vm)
        tuning.append(dict(method='logistic',C=c,**agg))
        if best is None or agg['ndcg1000']>best[0]:best=agg['ndcg1000'],scores
    predictions['logistic']=best[1]
    metrics,rows_by_method={},{}
    for name,scores in predictions.items():
        metrics[name],rows_by_method[name]=evaluate(scores,vg,vm,True)
    for name in vm[0]['standalone']:
        rows=[]
        for q in vm:
            pred=q['standalone'][name]
            rows.append(dict(qid=q['qid'],seed=q['seed'],predicted=pred,truth=q['truth'],new_truth=q['new_truth'],
                candidate_recall=len(set(pred)&set(q['truth']))/len(q['truth']),**rank_metrics(pred,q['truth'],q['new_truth'])))
        rows_by_method[name]=rows
        metrics[name]={key:float(np.mean([r[key] for r in rows if r[key] is not None]))
            if any(r[key] is not None for r in rows) else None for key in
            ['ndcg10','ndcg1000','rprecision','recall10','recall50','new_recall10','candidate_recall']}
    for name,rows in rows_by_method.items():
        (run/f'{name}.predictions.json').write_text(json.dumps(rows))
    report=dict(status='Development only; official test sealed',config=config,features=FEATURES,
        training_positive_injections=sum(m['training_positive_injections'] for m in tm),
        metrics=metrics,tuning=tuning,autoresearch_assumptions=ASSUMPTIONS,
        paired_validation_ndcg10={base:cluster_difference(rows_by_method['lambdamart'],rows_by_method[base])
            for base in ['frequency10','autoresearch_description','binary_lightgbm']},
        wall_seconds=time.perf_counter()-began,
        limitations=['MiniLM is a general pretrained semantic encoder, not a scholarly-specific encoder; pretraining contamination is not ruled out.',
            'Autoresearch implementation follows an incomplete published description.',
            'Standalone frequency uses deterministic ID ties and pre-date padding, not byte-exact v1 padding.',
            'Ablations hold the joint retrieval pool fixed; without_text removes reranker text only.',
            'One development seed; validation-selected scores and CIs are exploratory.'])
    (run/'report.json').write_text(json.dumps(report,indent=2))
    log(json.dumps(report,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
