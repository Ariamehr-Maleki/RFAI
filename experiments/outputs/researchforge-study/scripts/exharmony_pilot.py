"""Adapted author-disjoint DEVELOPMENT pilot. Test qrels are filtered out first."""
import argparse
import hashlib
import json
import math
import time
import collections
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from lightgbm import LGBMClassifier,LGBMRanker
from exharmony_index import INDEX,DATA,ROOT,AUDIT

OUT=ROOT/'outputs/researchforge-study'
FEATURES=['max_paper_cosine','sum_retrieved_cosine','mean_retrieved_cosine','retrieved_paper_count',
          'reciprocal_paper_rank_sum','mean_all_profile_cosine','log_profile_papers']

def metrics(pred,relevant):
    rel=np.array([a in relevant for a in pred],dtype=float);r=len(relevant)
    result={}
    for k in [10,20,100]:
        s=rel[:k];ranks=np.arange(1,len(s)+1)
        ideal=np.sum(1/np.log2(np.arange(min(k,r))+2))
        result[f'ndcg{k}']=float(np.sum(s/np.log2(ranks+1))/ideal) if ideal else 0.
        # trec_eval map_cut divides by ALL relevant items, not min(R,k).
        result[f'map{k}']=float(np.sum(np.cumsum(s)/ranks*s)/r) if r else 0.
        result[f'recall{k}']=float(s.sum()/r) if r else 0.
        hit=np.flatnonzero(s)
        result[f'rr{k}']=float(1/(hit[0]+1)) if len(hit) else 0.
    return result

def load_qrels(name,allowed):
    result=collections.defaultdict(set)
    with (DATA/'qrels'/f'qrels.test.{name}.tsv').open(encoding='utf-8') as f:
        for line in f:
            qid,rest=line.rstrip().split(None,1)
            if qid not in allowed:continue  # held-out relevance fields not parsed/stored
            _,aid,rel=rest.split()
            if int(rel)>0:result[qid].add(int(aid.rsplit('A',1)[-1]))
    return result

def evaluate(scores,groups,meta,qrels):
    rows=[];start=0
    for n,q in zip(groups,meta):
        pred=[q['candidates'][i] for i in np.argsort(-scores[start:start+n],kind='stable')[:100]]
        start+=n
        truth=qrels.get(q['qid'],set())
        if not truth:continue
        rows.append(dict(qid=q['qid'],predicted=pred,truth=sorted(truth),
            candidate_recall=len(set(q['candidates'])&truth)/len(truth),**metrics(pred,truth)))
    keys=[k for k in rows[0] if k not in ['qid','predicted','truth']] if rows else []
    return {k:float(np.mean([r[k] for r in rows])) for k in keys},rows

class Retriever:
    def __init__(self):
        manifest=json.loads((INDEX/'manifest.json').read_text())
        self.docs=sparse.csr_matrix((np.memmap(INDEX/'data.bin',dtype=np.float32,mode='r'),
            np.memmap(INDEX/'indices.bin',dtype=np.int32,mode='r'),np.load(INDEX/'indptr.npy',mmap_mode='r')),
            shape=(manifest['rows'],manifest['dimensions']),copy=False)
        self.pa=sparse.load_npz(INDEX/'paper_authors.npz')
        self.ap=sparse.load_npz(INDEX/'author_papers.npz')
        self.ids=np.load(INDEX/'author_ids.npy')
        self.order=np.argsort(self.ids);self.sorted_ids=self.ids[self.order]
        self.counts=np.diff(self.ap.indptr)
        self.vectorizer=HashingVectorizer(n_features=2**18,alternate_sign=False,norm='l2',stop_words='english',dtype=np.float32)

    def query(self,text,truth,k,training):
        vector=self.vectorizer.transform([text]).toarray().ravel()
        scores=self.docs@vector
        top_n=min(2000,len(scores))
        top=np.argpartition(-scores,top_n-1)[:top_n]
        top=sorted(top,key=lambda i:(-scores[i],int(i)))
        totals,maxima,counts,rrf=collections.Counter(),collections.Counter(),collections.Counter(),collections.Counter()
        for rank,p in enumerate(top):
            if scores[p]<=0:continue
            for a in self.pa.indices[self.pa.indptr[p]:self.pa.indptr[p+1]]:
                totals[a]+=float(scores[p]);maxima[a]=max(maxima[a],float(scores[p]));counts[a]+=1;rrf[a]+=1/(61+rank)
        candidates=sorted(totals,key=lambda a:(-maxima[a],-totals[a],int(self.ids[a])))[:k]
        retrieved_ids={int(self.ids[a]) for a in candidates}
        recall=len(retrieved_ids&truth)/len(truth) if truth else 0
        injected=0;unavailable=0
        if training:
            for aid in sorted(truth-retrieved_ids):
                loc=np.searchsorted(self.sorted_ids,aid)
                if loc<len(self.ids) and self.sorted_ids[loc]==aid:
                    row=int(self.order[loc])
                    if self.counts[row]>0:candidates.append(row);injected+=1
                    else:unavailable+=1
                else:unavailable+=1
        candidates.sort(key=lambda a:int(self.ids[a]))
        all_sums=self.ap[candidates]@scores
        rows=[]
        for i,a in enumerate(candidates):
            # Compute max across history for training-injected positives too, so
            # absence from retrieval does not create artificial zero text features.
            profile_scores=scores[self.ap.indices[self.ap.indptr[a]:self.ap.indptr[a+1]]]
            rows.append([float(profile_scores.max()) if len(profile_scores) else 0,totals[a],
                totals[a]/max(1,counts[a]),counts[a],rrf[a],float(all_sums[i])/max(1,self.counts[a]),math.log1p(self.counts[a])])
        ids=[int(self.ids[a]) for a in candidates]
        return np.array(rows,dtype=np.float32),np.array([a in truth for a in ids],dtype=np.int8),dict(candidates=ids,candidate_recall=recall,injected=injected,unavailable_training_positives=unavailable)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-queries',type=int,default=200);ap.add_argument('--valid-queries',type=int,default=100)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--candidates',type=int,default=1000);args=ap.parse_args()
    start=time.perf_counter();split=json.loads((AUDIT/'exharmony_split.json').read_text())
    run=OUT/'results'/f'exharmony-pilot-{args.train_queries}-{args.valid_queries}-{args.candidates}-s{args.seed}';run.mkdir(parents=True,exist_ok=True)
    config=dict(**vars(args),split_sha256=hashlib.sha256((AUDIT/'exharmony_split.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),selection='authors validation ndcg10')
    if (run/'config.json').exists():assert json.loads((run/'config.json').read_text())==config
    else:(run/'config.json').write_text(json.dumps(config,indent=2))
    rng=np.random.default_rng(args.seed)
    selected={k:sorted(rng.choice(split[k],min(n,len(split[k])),replace=False).tolist()) for k,n in [('train',args.train_queries),('validation',args.valid_queries)]}
    assert not (set(selected['train'])|set(selected['validation']))&set(split['test'])
    allowed=set(selected['train'])|set(selected['validation'])
    qrels=load_qrels('authors',allowed);texts={}
    with (DATA/'papers_test.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['id'] in allowed:texts[r['id']]=(r.get('title') or '')+' '+(r.get('abstract') or '')
    engine=Retriever();datasets={}
    for part in ['train','validation']:
        xs,ys,gs,meta=[],[],[],[];folder=run/part;folder.mkdir(exist_ok=True)
        for i,qid in enumerate(selected[part]):
            name=qid.rsplit('/',1)[-1];f=folder/(name+'.npz');j=folder/(name+'.json')
            if f.exists() and j.exists():
                with np.load(f) as z:x,y=z['X'],z['y']
                m=json.loads(j.read_text())
            else:
                x,y,m=engine.query(texts[qid],qrels.get(qid,set()),args.candidates,part=='train');m['qid']=qid
                np.savez_compressed(f,X=x,y=y);j.write_text(json.dumps(m))
            xs.append(x);ys.append(y);gs.append(len(y));meta.append(m)
            print(f'{part} {i+1}/{len(selected[part])}; recall={m["candidate_recall"]:.3f}',flush=True)
        datasets[part]=np.concatenate(xs),np.concatenate(ys),np.array(gs),meta
    tx,ty,tg,tm=datasets['train'];vx,vy,vg,vm=datasets['validation']
    predictions={'max_cosine':vx[:,0],'sum_cosine':vx[:,1],'mean_cosine':vx[:,2],'paper_rank_rrf':vx[:,4],'profile_mean_cosine':vx[:,5]}
    tuning=[]
    for name,cls in [('lambdamart',LGBMRanker),('binary_lightgbm',LGBMClassifier)]:
        best=None
        for leaves,trees in [(7,60),(15,100),(31,100)]:
            model=cls(n_estimators=trees,num_leaves=leaves,learning_rate=.05,reg_lambda=1,n_jobs=2,verbosity=-1,
                deterministic=True,force_col_wise=True,random_state=args.seed,objective='lambdarank' if name=='lambdamart' else 'binary')
            model.fit(tx,ty,**({'group':tg} if name=='lambdamart' else {}))
            s=model.predict(vx) if name=='lambdamart' else model.predict_proba(vx)[:,1]
            agg,_=evaluate(s,vg,vm,qrels);tuning.append(dict(method=name,leaves=leaves,trees=trees,**agg))
            if best is None or agg['ndcg10']>best[0]:best=agg['ndcg10'],s,model
        predictions[name]=best[1];best[2].booster_.save_model(str(run/(name+'.txt')))
    results={}
    for judgment in ['authors','cite','simcite']:
        labels=qrels if judgment=='authors' else load_qrels(judgment,set(selected['validation']))
        results[judgment]={}
        for name,s in predictions.items():
            agg,rows=evaluate(s,vg,vm,labels);results[judgment][name]=dict(judged_queries=len(rows),**agg)
            (run/f'{judgment}.{name}.predictions.json').write_text(json.dumps(rows))
    report=dict(status='Development only; adapted split; no held-out test relevance parsed',config=config,features=FEATURES,
        metrics=results,tuning=tuning,wall_seconds=time.perf_counter()-start,training_positive_injections=sum(q['injected'] for q in tm),
        unavailable_training_positives=sum(q['unavailable_training_positives'] for q in tm),
        limitations=['Lexical hashing, not BM25 or scholarly pretrained embeddings.',
            'Trained on Authors proxy only; Cite/SimCite are secondary transfer evaluations.',
            'Candidate pool selected by maximum paper similarity; fixed for learned-method comparison.',
            'Title-matching corpus documents removed; results not directly comparable to published unfiltered runs.',
            'One sampled development seed; no superiority or novelty claim.'])
    (run/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
