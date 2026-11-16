"""Globally gated frozen expert retrieval, no fit or test-positive injection."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
from threadpoolctl import threadpool_limits
import exharmony_bm25_pilot as bm
from exharmony_prepare import norm
from final_evaluation_gate import authorize_evaluation,local_file,canonical
from final_statistics import seed_average_pairs,cluster_inference,holm
from frozen_ranker import FrozenRanker,digest
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3]
CODE=['exharmony_final_run.py','exharmony_bm25_pilot.py','exharmony_index.py','exharmony_prepare.py',
      'hashed_bm25.py','final_evaluation_gate.py','final_statistics.py','frozen_ranker.py','supervisor_state.py']
INDEX_FILES=['manifest.json','data.bin','indices.bin','indptr.npy','paper_authors.npz','author_papers.npz',
             'author_ids.npy','bm25_stats.json','bm25_stats.npz']
JUDGMENTS=['authors','cite','simcite']


def read(path):return json.loads(Path(path).read_text())


def prepare(root,config):
    roles={r:{local_file(root,e['path']) for e in config['artifacts'] if e['role']==r}
           for r in ['code','data','model','development_report','split']}
    for name in CODE:
        if Path(__file__).with_name(name).resolve() not in roles['code']:raise ValueError('Unlocked source: '+name)
    if config.get('runner_schema')!=1 or config.get('feature_names')!=bm.FEATURES:raise ValueError('Frozen feature schema required')
    if type(config.get('candidates')) is not int or config['candidates']<1:raise ValueError('Frozen positive candidate count required')
    sampling=config.get('query_sampling',{});inference=config.get('inference',{})
    if set(sampling)!={'count','seed'} or type(sampling['count']) is not int or sampling['count']<0 or type(sampling['seed']) is not int:
        raise ValueError('Freeze query census or deterministic sample')
    if set(inference)!={'repetitions','seed'} or type(inference['repetitions']) is not int or inference['repetitions']<100 or type(inference['seed']) is not int:
        raise ValueError('Freeze paired inference settings')
    if config.get('primary_judgment')!='authors':raise ValueError('Authors is the primary judgment')
    inputs=config.get('inputs',{})
    if set(inputs)!={'queries','judgeable','split','authors','cite','simcite'}:raise ValueError('Explicit query metadata, split and judgments required')
    paths={k:local_file(root,v) for k,v in inputs.items()}
    if paths['split'] not in roles['split'] or not {v for k,v in paths.items() if k!='split'}<=roles['data']:
        raise ValueError('Unbound exHarmony inputs')
    index=(root/config['index_directory']).resolve()
    if Path(config['index_directory']).is_absolute() or not index.is_relative_to(root):raise ValueError('Invalid index directory')
    if not {index/name for name in INDEX_FILES}<=roles['data']:raise ValueError('Unbound index files')
    meta=read(index/'bm25_stats.json')
    if meta['index_manifest_sha256']!=digest(index/'manifest.json') or meta['stats_sha256']!=digest(index/'bm25_stats.npz'):
        raise ValueError('Index statistics mismatch')
    models={}
    for name,method in config['methods'].items():
        if not re.fullmatch(r'[A-Za-z0-9_+.\-]+',name):raise ValueError('Unsafe method name')
        if method.get('kind')=='feature':
            if method.get('feature') not in bm.FEATURES:raise ValueError('Unknown fixed feature')
        elif method.get('kind')=='learned':
            if set(method['models'])!=set(map(str,config['seeds'])):raise ValueError('Missing model seeds')
            for seed in config['seeds']:
                spec=dict(method['models'][str(seed)]);path=local_file(root,spec['path']);source=local_file(root,spec['source_report'])
                if path not in roles['model'] or source not in roles['development_report']:raise ValueError('Unbound model provenance')
                if spec['features']!=bm.FEATURES or spec['kind']!='lightgbm':raise ValueError('Unknown expert model schema')
                if digest(source)!=spec['source_report_sha256']:raise ValueError('Changed model provenance')
                spec['path']=str(path);models[name,seed]=FrozenRanker(spec,bm.FEATURES)
        else:raise ValueError('Unknown expert method')
    return paths,index,models


def query_metadata(paths,allowed):
    """All held-out components, including bridges outside any sampled subset."""
    parent={q:q for q in allowed};texts={};first_title={};first_author={}
    def root(q):
        while parent[q]!=q:parent[q]=parent[parent[q]];q=parent[q]
        return q
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    with paths['queries'].open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);qid=row['id']
            if qid not in allowed:continue
            if qid in texts:raise ValueError('Duplicate query ID')
            texts[qid]=(row.get('title') or '')+' '+(row.get('abstract') or '')
            title=norm(row.get('title'))
            if title:
                if title in first_title:union(qid,first_title[title])
                else:first_title[title]=qid
    with paths['judgeable'].open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);qid=row['id']
            if qid not in allowed:continue
            for entry in row.get('authorships',[]):
                author=entry.get('author',{}).get('id')
                if not author:continue
                if author in first_author:union(qid,first_author[author])
                else:first_author[author]=qid
    if set(texts)!=allowed:raise ValueError('Missing held-out query text')
    return texts,{q:root(q) for q in allowed}


def labels(path,allowed):
    result={q:set() for q in allowed}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            qid,rest=line.rstrip().split(None,1)
            if qid not in allowed:continue
            _,author,rel=rest.split()
            if int(rel)>0:result[qid].add(int(author.rsplit('A',1)[-1]))
    return result


def engine_at(index):
    # The existing development class uses one module-global index path.
    # Bind it only during construction; this runner is a single-writer process.
    previous=bm.INDEX;bm.INDEX=index
    try:return bm.Retriever()
    finally:bm.INDEX=previous


def verify_rows(rows):
    import pytrec_eval
    qrels={r['qid']:{str(a):1 for a in r['truth']} for r in rows}
    rankings={r['qid']:{str(a):float(len(r['predicted'])-i) for i,a in enumerate(r['predicted'])} for r in rows}
    requested={f'{m}.{k}' for m in ['ndcg_cut','map_cut','recall'] for k in [10,20,100]}
    official=pytrec_eval.RelevanceEvaluator(qrels,requested).evaluate(rankings);errors=[]
    for k in [10,20,100]:
        cut={r['qid']:{str(a):float(len(r['predicted'])-i) for i,a in enumerate(r['predicted'][:k])} for r in rows}
        rr=pytrec_eval.RelevanceEvaluator(qrels,{'recip_rank'}).evaluate(cut)
        for r in rows:
            for own,ref in [('ndcg','ndcg_cut'),('map','map_cut'),('recall','recall')]:errors.append(abs(r[f'{own}{k}']-official[r['qid']][f'{ref}_{k}']))
            errors.append(abs(r[f'rr{k}']-rr[r['qid']]['recip_rank']))
    maximum=max(errors,default=0.)
    if maximum>1e-10:raise ValueError('Independent TREC metric mismatch')
    return dict(queries=len(rows),comparisons=len(errors),max_absolute_error=maximum)


def run(root,lock_path,run_id,ledger):
    root=Path(root).resolve();cfg=authorize_evaluation(root,lock_path,'exHarmony',run_id,ledger)
    paths,index,models=prepare(root,cfg)
    out=(root/cfg['output_directory']).resolve()
    if Path(cfg['output_directory']).is_absolute() or not out.is_relative_to(root) or out==root:raise ValueError('Invalid output directory')
    identity=dict(run_id=run_id,lock_sha256=read(lock_path)['lock_sha256'],config_sha256=hashlib.sha256(canonical(cfg)).hexdigest())
    out.mkdir(parents=True,exist_ok=True);writer=out/'writer.lock'
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Changed final run identity')
    fd=os.open(writer,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid()),stream)
    try:
        atomic_state(out/'identity.json',identity)
        if (out/'report.json').exists():
            report=read(out/'report.json')
            for name,sha in report['output_sha256'].items():
                if digest(out/name)!=sha:raise ValueError('Completed expert output changed')
            return report
        split=read(paths['split']);allowed=set(split['test'])
        if len(allowed)!=len(split['test']) or allowed&(set(split['train'])|set(split['validation'])):raise ValueError('Invalid disjoint split')
        # All label/metadata loading follows global authorization and input bindings.
        texts,clusters=query_metadata(paths,allowed)
        count=cfg['query_sampling']['count']
        if count>len(allowed):raise ValueError('Frozen query count exceeds test size')
        selected=sorted(allowed) if count==0 else sorted(np.random.default_rng(cfg['query_sampling']['seed']).choice(sorted(allowed),count,replace=False).tolist())
        truths={j:labels(paths[j],set(selected)) for j in JUDGMENTS}
        engine=engine_at(index);query_dir=out/'queries';query_dir.mkdir(exist_ok=True)
        inventory={'identity.json':digest(out/'identity.json')};query_paths=[]
        with threadpool_limits(limits=1):
            for number,qid in enumerate(selected):
                stem=hashlib.sha256(qid.encode()).hexdigest();path=query_dir/(stem+'.json');receipt=query_dir/(stem+'.receipt');feature=query_dir/(stem+'.npz')
                if receipt.exists():
                    record=read(receipt)
                    if any(record.get(k)!=v for k,v in identity.items()) or record['sha256']!=digest(path) or record['features_sha256']!=digest(feature):raise ValueError('Committed expert query changed')
                else:
                    # Pass no truth whatsoever to candidate generation.
                    x,_,meta=engine.query(texts[qid],set(),cfg['candidates'],False)
                    x=np.asarray(x,dtype=np.float32).reshape(-1,len(bm.FEATURES))
                    if meta['injected'] or len(meta['candidates'])!=len(x) or len(set(meta['candidates']))!=len(x):raise ValueError('Invalid candidates/injection')
                    if not np.isfinite(x).all():raise ValueError('Nonfinite expert features')
                    predictions={}
                    for name,method in cfg['methods'].items():
                        predictions[name]={}
                        for seed in cfg['seeds']:
                            scores=np.empty(0) if not len(x) else models[name,seed].predict(x) if method['kind']=='learned' else x[:,bm.FEATURES.index(method['feature'])]
                            order=np.argsort(-scores,kind='stable')[:100]
                            predictions[name][str(seed)]=[meta['candidates'][i] for i in order]
                    with feature.with_suffix('.part').open('wb') as stream:np.savez_compressed(stream,X=x,candidates=np.asarray(meta['candidates'],dtype=np.int64))
                    feature.with_suffix('.part').replace(feature)
                    atomic_state(path,dict(qid=qid,seed=clusters[qid],truth={j:sorted(truths[j][qid]) for j in JUDGMENTS},candidates=meta['candidates'],predictions=predictions,injected=0))
                    atomic_state(receipt,dict(**identity,sha256=digest(path),features_sha256=digest(feature)))
                query_paths.append(path)
                for p in [path,receipt,feature]:inventory[p.relative_to(out).as_posix()]=digest(p)
                atomic_state(out/'progress.json',dict(**identity,worker_pid=os.getpid(),completed_queries=number+1,total_queries=len(selected)))
        del engine;gc.collect()
        stored=[read(path) for path in query_paths];metrics={};verification={};pairs={}
        for judgment in JUDGMENTS:
            metrics[judgment]={};all_rows={}
            for name in cfg['methods']:
                metrics[judgment][name]={};all_rows[name]={}
                for seed in cfg['seeds']:
                    rows=[]
                    for q in stored:
                        truth=set(q['truth'][judgment])
                        if not truth:continue
                        pred=q['predictions'][name][str(seed)]
                        rows.append(dict(qid=q['qid'],seed=q['seed'],truth=sorted(truth),predicted=pred,
                            candidate_recall=len(set(q['candidates'])&truth)/len(truth),**bm.metrics(pred,truth)))
                    if not rows:raise ValueError('No judged queries for '+judgment)
                    file=out/f'{judgment}.{name}.s{seed}.predictions.json';atomic_state(file,rows)
                    verification[file.name]=verify_rows(read(file));inventory[file.name]=digest(file);all_rows[name][seed]=rows
                    metrics[judgment][name][str(seed)]={k:float(np.mean([r[k] for r in rows])) for k in ['candidate_recall']+[f'{m}{k}' for m in ['ndcg','map','recall','rr'] for k in [10,20,100]]}
            # Primary family only Authors; other judgments are descriptive transfer.
            if judgment==cfg['primary_judgment']:
                for a,b in cfg['primary_comparisons']:
                    paired=seed_average_pairs(all_rows[a],all_rows[b],'ndcg10')
                    key=a+'__minus__'+b
                    if key in pairs:raise ValueError('Duplicate primary comparison')
                    pairs[key]=dict(**cluster_inference(paired['differences'],paired['authors'],**cfg['inference']),per_seed_deltas=paired['per_seed_deltas'])
        adjusted=holm({k:v['p_two_sided_cluster_signflip'] for k,v in pairs.items()})
        for name,value in pairs.items():value['holm_p']=adjusted[name]
        atomic_state(out/'metric_verification.json',dict(status='PASS',library='pytrec_eval',results=verification));inventory['metric_verification.json']=digest(out/'metric_verification.json')
        summary={j:{m:{k:dict(mean=float(np.mean([v[k] for v in seeds.values()])),sample_sd=float(np.std([v[k] for v in seeds.values()],ddof=1))) for k in next(iter(seeds.values()))} for m,seeds in methods.items()} for j,methods in metrics.items()}
        report=dict(status='Completed frozen adapted exHarmony evaluation',**identity,config=cfg,queries=len(selected),query_ids=selected,
            judged_queries={j:sum(bool(q['truth'][j]) for q in stored) for j in JUDGMENTS},test_components=len(set(clusters.values())),
            per_seed=metrics,seed_summary=summary,primary_comparisons=pairs,output_sha256=inventory,
            limitations=['Adapted author/title-component split, not published benchmark protocol.',
                'Authors proxy is primary; Cite/SimCite are descriptive transfer, not collaboration outcomes.',
                'Raw relevance includes missing author profiles; zero retrieval is retained.',
                'Undated filtered corpus; temporal provenance cannot be established.',
                'Hashed BM25 is not exact Lucene/Pyserini; query-author components omit shared cited-expert dependence.',
                'Fixed baselines repeated across seeds do not add independent observations.'])
        atomic_state(out/'report.json',report);return report
    finally:writer.unlink()


if __name__=='__main__':
    import psutil
    parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True,type=Path);parser.add_argument('--run-id',required=True);parser.add_argument('--ledger',required=True,type=Path);args=parser.parse_args()
    if psutil.virtual_memory().available<3500*1024**2:raise RuntimeError('Wait for3500MiB availableRAM')
    print(json.dumps(run(ROOT,args.lock,args.run_id,args.ledger),indent=2))
