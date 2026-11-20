"""Resumable, pre-frozen additional-query check. Never modifies original artifacts."""
from pathlib import Path
import sys,os,json,hashlib,time,gc,argparse,shutil,uuid
from datetime import date,timedelta,datetime,timezone
import numpy as np
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits

R=Path(__file__).resolve().parent;S=R.parent;ROOT=S.parents[1];OUT=R/'fresh-cohort';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(S/'scripts'))
from prescience_pilot import day,unique_author_ids,make_queries,rank_metrics
from prescience_hybrid_graph import QueryEngine,FEATURES
from text_index import TextIndex
from semantic_index import SemanticIndex
from frozen_ranker import FrozenRanker
from final_statistics import cluster_inference,holm

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.partial.'+str(os.getpid())+'.'+uuid.uuid4().hex)
 try:
  t.write_text(json.dumps(o,indent=2),encoding='utf-8')
  for attempt in range(12):
   try:
    t.replace(p);return
   except PermissionError:
    if attempt==11:raise
    time.sleep(min(0.05*2**attempt,1.0))
 finally:
  t.unlink(missing_ok=True)
def freeze(p,o):
 if p.exists():assert read(p)==o,'Frozen manifest changed: '+str(p)
 else:write(p,o)
def memory_gate(mib,phase):
 start=time.monotonic()
 while psutil.virtual_memory().available<mib*1024**2:
  write(OUT/'resource-status.json',{'phase':phase,'pid':os.getpid(),'required_MiB':mib,'available_MiB':psutil.virtual_memory().available/1024**2,'status':'waiting; no scientific setting changed'})
  print('resource wait',phase,flush=True);time.sleep(15)
  if time.monotonic()-start>=300:raise RuntimeError('Resource wait exceeded 300 seconds; resumable without changing settings')
 if shutil.disk_usage(OUT).free<1024**3:raise RuntimeError('Disk reserve reached')

def prelock():
 lock=read(S/'final/global-lock.json');cfg=lock['payload']['plan']['datasets']['prescience']
 assets={}
 for art in cfg['artifacts']:
  path=art['path'];expected=lock['payload']['artifacts'][path]['sha256'];actual=sha(ROOT/path)
  assert actual==expected,'Original locked asset changed: '+path;assets[path]=actual
 dependencies=[R/'FRESH_COHORT_PROTOCOL.md',R/'RESOURCE_ADDENDUM_V2.md',R/'RESOURCE_ADDENDUM_V3.md',R/'fresh-resource-probe.json',Path(__file__),S/'final/global-lock.json',S/'results/prescience-expanded-1200-1000-500-1000/query_manifest.json']
 dependencies+=list((S/'review-20261107').glob('official_comparator*.py'))
 dependencies+=list((S/'review-20261107/upstream').rglob('*.py'))
 for p in dependencies:assets[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
 spec={'design':'Additional disjoint-query sample; known original outcomes; no refitting','date':'2026-11-20','count':1000,'sampling_seed':20261120,'padding_seed':42,'model_seeds':[42,43,44],'original_lock_sha256':lock['lock_sha256'],'resume_from_lock_v2_sha256':sha(OUT/'pre-inference-lock-v2.json'),'frozen_sample_sha256':sha(OUT/'sample-manifest.json'),'assets':assets,'methods':['full_lambdamart','full_binary_lightgbm','official_harmonized'],'primary':'full_lambdamart minus official_harmonized','secondary':'full_lambdamart minus full_binary_lightgbm','inference':{'repetitions':10000,'seed':314159,'holm_family_size':2}}
 freeze(OUT/'pre-inference-lock-v3.json',spec);return cfg

def load_slim(cfg):
 papers={}
 for batch in pq.ParquetFile(ROOT/cfg['corpus_inputs']['corpus']).iter_batches(batch_size=2048,columns=['corpus_id','date','roles','authors','categories','key_references']):
  for row in batch.to_pylist():
   cid=sys.intern(str(row['corpus_id']));assert cid not in papers
   if not row.get('date'):continue
   papers[cid]={'date':day(row['date']),'authors':unique_author_ids(row['authors']),'target':'target' in (row['roles'] or []),'categories':tuple(sys.intern(c) for c in row['categories'] or []),'refs':tuple(sys.intern(str(v['corpus_id'])) for v in row['key_references'] or [] if v['corpus_id'])}
 histories={};dates={}
 with (ROOT/cfg['corpus_inputs']['author_history']).open(encoding='utf-8') as f:
  for line in f:
   row=json.loads(line);a=sys.intern(str(row['key']));ids=sorted({sys.intern(str(i)) for i in row['value'] or [] if str(i) in papers},key=lambda i:(papers[i]['date'],i))
   if ids:histories[a]=ids;dates[a]=[papers[i]['date'] for i in ids]
 return papers,histories,dates

def rankers():
 cfg=prelock();memory_gate(3000,'before-ranker-load');papers,histories,dates=load_slim(cfg)
 eligible=make_queries(papers,histories,dates);assert len(eligible)==47997
 original=read(S/'results/final/prescience/full_lambdamart.s42.predictions.json');exclude={q['qid'] for q in original}
 development=read(S/'results/prescience-expanded-1200-1000-500-1000/query_manifest.json')
 def collect(x):
  if isinstance(x,dict):
   if 'qid' in x:exclude.add(str(x['qid']))
   for v in x.values():collect(v)
  elif isinstance(x,list):
   for v in x:collect(v)
 collect(development)
 remaining=sorted([q for q in eligible if q['qid'] not in exclude],key=lambda q:q['qid'])
 chosen=np.random.default_rng(20261120).choice(len(remaining),1000,replace=False)
 queries=sorted([remaining[i] for i in chosen],key=lambda q:(q['date'],q['qid']))
 from collections import Counter
 quarter=lambda q: str((date(1970,1,1)+timedelta(days=q['date'])).year)+'-Q'+str(((date(1970,1,1)+timedelta(days=q['date'])).month-1)//3+1)
 manifest={'prelock_sha256':sha(OUT/'pre-inference-lock-v2.json'),'eligible':len(eligible),'remaining':len(remaining),'queries':[{k:q[k] for k in ['qid','seed','date']} for q in queries],
  'original_query_overlap':len(set(q['qid'] for q in queries)&set(q['qid'] for q in original)),
  'original_seed_author_overlap':len(set(q['seed'] for q in queries)&set(q['seed'] for q in original)),
  'selected_quarters':dict(Counter(map(quarter,queries))),'remaining_quarters':dict(Counter(map(quarter,remaining)))}
 assert manifest['original_query_overlap']==0;freeze(OUT/'sample-manifest.json',manifest)
 del eligible,remaining,development;gc.collect()
 lexical=TextIndex(ROOT/cfg['index_inputs']['lexical']);semantic=SemanticIndex(ROOT/cfg['index_inputs']['semantic'],ROOT/cfg['index_inputs']['lexical'])
 engine=QueryEngine(papers,histories,dates,lexical,semantic);models={}
 for name in ['full_lambdamart','full_binary_lightgbm']:
  for seed,spec in cfg['methods'][name]['models'].items():
   spec=dict(spec);spec['path']=str(ROOT/spec['path']);models[name,seed]=FrozenRanker(spec,FEATURES)
 identity=sha(OUT/'sample-manifest.json')
 with threadpool_limits(limits=1):
  parity=[]
  for old in sorted(original,key=lambda q:q['qid'])[:5]:
   q={k:old[k] for k in ['qid','seed','date','truth']};X,y,m=engine.run(q,1000,42,False);key=hashlib.sha256(q['qid'].encode()).hexdigest();folder=S/'results/final/prescience/queries'
   saved=read(folder/(key+'.json'))
   with np.load(folder/(key+'.full.npz'),allow_pickle=False) as cache:assert np.array_equal(X,cache['X']) and np.array_equal(y,cache['y'])
   assert m['candidates']==saved['metadata']['full']['candidates'];parity.append(q['qid'])
  write(OUT/'original-feature-parity.json',{'status':'PASS','queries':parity,'feature_matrix':'exact float32 equality','candidate_order':'exact equality','labels':'exact equality'})
  for i,q in enumerate(queries):
   destination=OUT/'ranker-queries'/(q['qid']+'.json')
   if destination.exists():
    saved=read(destination);assert saved['sample_sha256']==identity
    assert sha(destination)==read(destination.with_suffix('.receipt.json'))['sha256']
    assert sha(destination.with_suffix('.npz'))==saved['cache_sha256'];continue
   memory_gate(640,'between-ranker-queries');start=time.perf_counter();X,y,m=engine.run(q,1000,42,False)
   assert m['training_positive_injections']==0;ids=m['candidates'];predictions={}
   for (name,seed),model in models.items():
    score=model.predict(X);pred=[ids[j] for j in sorted(range(len(ids)),key=lambda j:(-score[j],ids[j]))]
    new=set(m['new_truth']);repeat=set(q['truth'])-new
    predictions[name+'.s'+seed]={'predicted':pred,**rank_metrics(pred,q['truth'],new),'repeat_recall10':len(set(pred[:10])&repeat)/len(repeat) if repeat else None}
   cache=destination.with_suffix('.npz');cache.parent.mkdir(exist_ok=True)
   with cache.open('wb') as f:np.savez_compressed(f,X=X,y=y)
   write(destination,{'sample_sha256':identity,'query':q,'metadata':m,'predictions':predictions,'cache_sha256':sha(cache),'operational_seconds':time.perf_counter()-start})
   write(destination.with_suffix('.receipt.json'),{'sha256':sha(destination)})
   write(OUT/'progress.json',{'phase':'rankers','completed':i+1,'total':1000,'pid':os.getpid()})
   if (i+1)%20==0:print('rankers',i+1,'/1000',flush=True)
 write(OUT/'rankers-complete.json',{'queries':1000,'sample_sha256':identity})

def official():
 if os.environ.get('PYTHONHASHSEED')!='0':raise RuntimeError('Set PYTHONHASHSEED=0 before launch')
 prelock();manifest=read(OUT/'sample-manifest.json');identity=sha(OUT/'sample-manifest.json');assert read(OUT/'rankers-complete.json')['sample_sha256']==identity
 sys.path.insert(0,str(S/'review-20261107'));import official_comparator as base;import official_comparator_bounded as bounded
 memory_gate(1800,'before-official-load');papers,histories,clean,first,audit=base.load_inputs();env=bounded.factory()
 order=sorted(first.items(),key=lambda kv:kv[1]);eligible=set();cursor=0
 for i,q0 in enumerate(manifest['queries']):
  record=read(OUT/'ranker-queries'/(q0['qid']+'.json'));q=record['query'];m=record['metadata'];cutoff=(date(1970,1,1)+timedelta(days=q['date'])).isoformat()
  while cursor<len(order) and order[cursor][1]<cutoff:eligible.add(order[cursor][0]);cursor+=1
  path=OUT/'official-queries'/(q['qid']+'.json')
  if path.exists():
   assert read(path)['sample_sha256']==identity
   assert sha(path)==read(path.with_suffix('.receipt.json'))['sha256'];continue
  memory_gate(640,'between-official-queries');view=base.PaperView(papers,cutoff,historical=True,dedup=True);start=time.perf_counter()
  predicted,scores=env['predict_coauthors'](q['seed'],cutoff,1000,eligible,clean,view)
  assert len(predicted)==1000 and q['seed'] not in predicted and np.isfinite(scores).all()
  write(path,{'sample_sha256':identity,**q,'new_truth':m['new_truth'],'predicted':predicted,'scores':scores,
    **base.metrics(predicted,q['truth'],m['new_truth']),'operational_seconds':time.perf_counter()-start})
  write(path.with_suffix('.receipt.json'),{'sha256':sha(path)})
  view.cache.clear();del view
  write(OUT/'progress.json',{'phase':'official','completed':i+1,'total':1000,'pid':os.getpid()})
  if (i+1)%20==0:print('official',i+1,'/1000',flush=True)
 write(OUT/'official-complete.json',{'queries':1000,'sample_sha256':identity,'helper_calls':bounded.checks,'evictions':bounded.evictions})

def summarize():
 manifest=read(OUT/'sample-manifest.json');rows={m:[] for m in ['ranker','classifier','official']};groups=[];ids=[];per_seed={}
 for q in manifest['queries']:
  r=read(OUT/'ranker-queries'/(q['qid']+'.json'));o=read(OUT/'official-queries'/(q['qid']+'.json'));groups.append(q['seed']);ids.append(q['qid'])
  for out,method in [('ranker','full_lambdamart'),('classifier','full_binary_lightgbm')]:
   for seed in [42,43,44]:per_seed.setdefault(method+'.s'+str(seed),[]).append(r['predictions'][method+'.s'+str(seed)]['ndcg1000'])
   metrics={}
   for k in ['ndcg1000','recall10','new_recall10','repeat_recall10']:
    values=[r['predictions'][method+'.s'+str(seed)][k] for seed in [42,43,44]];metrics[k]=float(np.mean(values)) if values[0] is not None else None
   metrics['candidate_recall']=len(set(r['metadata']['candidates'])&set(r['query']['truth']))/len(set(r['query']['truth']));rows[out].append(metrics)
  rows['official'].append({k:o[k] for k in rows['ranker'][-1]})
 summaries={m:{k:float(np.mean([r[k] for r in records if r[k] is not None])) for k in records[0]} for m,records in rows.items()}
 contrasts={}
 for other in ['official','classifier']:
  d=np.array([a['ndcg1000']-b['ndcg1000'] for a,b in zip(rows['ranker'],rows[other])]);contrasts['ranker_minus_'+other]=cluster_inference(d,groups)
 adjusted=holm({k:v['p_two_sided_cluster_signflip'] for k,v in contrasts.items()})
 for k,v in contrasts.items():v['holm_p_two_contrasts']=adjusted[k]
 write(OUT/'results.json',{'status':'Complete additional disjoint-query sample; post-result robustness, not independent external replication','sample_sha256':sha(OUT/'sample-manifest.json'),'queries':len(ids),'seed_author_clusters':len(set(groups)),'summary':summaries,'per_seed_ndcg1000':{k:float(np.mean(v)) for k,v in per_seed.items()},'contrasts':contrasts,'limitations':['Same provider release and overlapping historical network; no independent dataset','No refitting or tuning','Operational durations are not isolated latency benchmarks']})
 print(json.dumps(summaries),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['rankers','official','summarize']);args=p.parse_args();globals()[args.stage]()
