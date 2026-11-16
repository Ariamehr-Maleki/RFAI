"""Frozen PreScience inference/evaluation; never tunes or fits on held-out data.

Requires a real global three-dataset lock. No usable default final configuration
is supplied. Synthetic integration tests create temporary fixture locks only.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from final_evaluation_gate import authorize_evaluation,local_file,canonical
from prescience_final_data import load_authorized_corpus
from prescience_hybrid_graph import QueryEngine,FEATURES
from prescience_retrieval_ablation import NoContentEngine,model_columns
from prescience_fixed_scorer import FixedScorer,VARIANTS
from prescience_pilot import rank_metrics
from semantic_index import SemanticIndex
from text_index import TextIndex
from frozen_ranker import FrozenRanker,digest
from final_statistics import seed_average_pairs,cluster_inference,holm
from verify_rank_predictions import verify
from supervisor_state import atomic_state
from final_subgroups import compare as subgroup_compare

ROOT=Path(__file__).resolve().parents[3]
CODE=['prescience_final_run.py','prescience_final_data.py','final_evaluation_gate.py',
      'prescience_hybrid_graph.py','prescience_retrieval_ablation.py','prescience_pilot.py',
      'prescience_fixed_scorer.py','semantic_index.py','text_index.py','frozen_ranker.py',
      'final_statistics.py','verify_rank_predictions.py','supervisor_state.py','publication_ablation_design.py',
      'final_subgroups.py','subgroup_analysis.py']


def read(path):return json.loads(Path(path).read_text())


def output_dir(root,relative):
    if Path(relative).is_absolute():raise ValueError('Output must be workspace relative')
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Output outside workspace')
    return path


def prepare(root,config):
    """Validate frozen inputs/model schemas before opening any corpus labels."""
    roles={role:{local_file(root,e['path']) for e in config['artifacts'] if e['role']==role}
           for role in ['code','model','data','development_report']}
    for name in CODE:
        if Path(__file__).with_name(name).resolve() not in roles['code']:raise ValueError('Runner dependency not locked: '+name)
    if config.get('runner_schema')!=1 or config.get('feature_names')!=FEATURES:raise ValueError('Unrecognized final schema')
    if type(config.get('candidates')) is not int or config['candidates']<1 or type(config.get('padding_seed')) is not int:
        raise ValueError('Freeze positive candidate count and padding seed')
    inference=config.get('inference',{})
    if set(inference)!={'repetitions','seed'} or type(inference['repetitions']) is not int or inference['repetitions']<100:
        raise ValueError('Freeze statistical repetitions and seed')
    if type(inference['seed']) is not int:raise ValueError('Integer inference seed required')
    subgroup_pairs=config.get('subgroup_comparisons')
    if not subgroup_pairs or any(len(p)!=2 or p[0]==p[1] or any(m not in config['methods'] for m in p) for p in subgroup_pairs):
        raise ValueError('Freeze secondary subgroup comparison pairs')
    modes=set();models={}
    for name,method in config['methods'].items():
        if not re.fullmatch(r'[A-Za-z0-9_+.\-]+',name):raise ValueError('Invalid method identifier')
        mode=method.get('candidate_mode')
        if mode not in ['full','no_content']:raise ValueError('Explicit candidate mode required')
        modes.add(mode);kind=method.get('kind')
        if kind=='learned':
            if set(method['models'])!=set(map(str,config['seeds'])):raise ValueError('Missing per-seed frozen models')
            for seed in config['seeds']:
                spec=dict(method['models'][str(seed)]);path=local_file(root,spec['path'])
                if not set(spec['features'])<={FEATURES[i] for i in model_columns(mode)}:
                    raise ValueError('Model features violate candidate-mode content policy')
                source=local_file(root,spec['source_report'])
                if path not in roles['model'] or source not in roles['development_report']:
                    raise ValueError('Model/development provenance not locked')
                if digest(source)!=spec['source_report_sha256']:raise ValueError('Development provenance changed')
                spec['path']=str(path);models[name,seed]=FrozenRanker(spec,FEATURES)
        elif kind=='feature':
            if method.get('feature') not in FEATURES:raise ValueError('Unknown baseline feature')
            if method['feature'] not in {FEATURES[i] for i in model_columns(mode)}:raise ValueError('Content feature in no-content mode')
        elif kind=='standalone':
            if method.get('ranking') not in ['frequency10_standalone','autoresearch_description_standalone',
                                           'lexical_standalone','semantic_standalone','rrf_standalone']:
                raise ValueError('Unknown standalone baseline')
            if mode=='no_content' and method['ranking'] in ['lexical_standalone','semantic_standalone']:
                raise ValueError('Content retrieval cannot be requested in no-content mode')
        elif kind=='fixed':
            if method.get('variant') not in VARIANTS:raise ValueError('Fixed variant must be selected before test')
        else:raise ValueError('Unknown frozen method kind')
    indices=None
    if 'full' in modes:
        paths={key:output_dir(root,config['index_inputs'][key]) for key in ['lexical','semantic']}
        for directory in paths.values():
            manifest=directory/'manifest.json'
            if manifest not in roles['data']:raise ValueError('Index manifest not locked')
        lexical=read(paths['lexical']/'manifest.json');semantic=read(paths['semantic']/'manifest.json')
        required=[paths['lexical']/name for name in lexical['files']]+[paths['semantic']/'embeddings.f16']
        if not set(required)<=roles['data']:raise ValueError('All consumed index files must be locked')
        corpus=local_file(root,config['corpus_inputs']['corpus']);source_sha=digest(corpus)
        if lexical['source_sha256']!=source_sha or semantic['config']['source_sha256']!=source_sha:
            raise ValueError('Final indices do not match explicit final corpus')
        indices=(TextIndex(paths['lexical']),SemanticIndex(paths['semantic'],paths['lexical']))
    return modes,models,indices


def prediction(q,metadata,ranking):
    truth=q['truth'];new=metadata['new_truth']
    return dict(qid=q['qid'],seed=q['seed'],date=q['date'],seed_history_count=metadata['seed_history_count'],
                truth=truth,new_truth=new,predicted=ranking,
                candidate_recall=len(set(ranking)&set(truth))/len(truth),**rank_metrics(ranking,truth,new))


def intern_prediction_rows(result):
    """Share equal author-ID strings after JSON resume; keep all values/order."""
    for seed_rows in result['predictions'].values():
        for row in seed_rows.values():
            for key in ['truth','new_truth','predicted']:
                row[key]=[sys.intern(value) for value in row[key]]
            row['qid']=sys.intern(row['qid']);row['seed']=sys.intern(row['seed'])
    return result


def run(root,lock_path,run_id,ledger_dir):
    root=Path(root).resolve()
    config=authorize_evaluation(root,lock_path,'prescience',run_id,ledger_dir)
    identity=dict(run_id=run_id,lock_sha256=read(lock_path)['lock_sha256'],config_sha256=hashlib.sha256(canonical(config)).hexdigest())
    modes,models,indices=prepare(root,config)
    out=output_dir(root,config['output_directory']);out.mkdir(parents=True,exist_ok=True)
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Final output belongs to another frozen run')
    writer=out/'writer.lock';fd=os.open(writer,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid()),stream)
    try:
        atomic_state(out/'identity.json',identity)
        if (out/'report.json').exists():
            report=read(out/'report.json')
            for name,sha in report['output_sha256'].items():
                if digest(out/name)!=sha:raise ValueError('Completed final output changed')
            return report
        _,papers,histories,dates,queries,audit=load_authorized_corpus(root,lock_path,run_id,ledger_dir)
        if not queries:raise ValueError('No eligible final queries')
        atomic_state(out/'data_audit.json',audit)
        engines={mode:QueryEngine(papers,histories,dates,*indices) if mode=='full' else NoContentEngine(papers,histories,dates) for mode in modes}
        predictions={name:{seed:[] for seed in config['seeds']} for name in config['methods']}
        receipts={};folder=out/'queries';folder.mkdir(exist_ok=True)
        with threadpool_limits(limits=1):
            for i,q in enumerate(queries):
                key=hashlib.sha256(str(q['qid']).encode()).hexdigest();path=folder/(key+'.json');receipt=path.with_suffix('.receipt')
                if receipt.exists():
                    saved=read(receipt)
                    for filename,sha in saved.items():
                        if digest(folder/filename)!=sha:raise ValueError('Committed final query artifact changed')
                    result=read(path)
                    if result['query']!=q:raise ValueError('Resumed final query changed')
                else:
                    result=dict(query=q,metadata={},predictions={});artifacts=[];matrices={}
                    for mode,engine in engines.items():
                        x,y,metadata=engine.run(q,config['candidates'],config['padding_seed'],False)
                        if metadata['training_positive_injections']!=0:raise ValueError('Held-out positive injection prohibited')
                        matrices[mode]=x;result['metadata'][mode]=metadata;feature_path=folder/(key+'.'+mode+'.npz')
                        with feature_path.open('wb') as stream:np.savez_compressed(stream,X=x,y=y)
                        artifacts.append(feature_path)
                    fixed={}
                    for name,method in config['methods'].items():
                        mode=method['candidate_mode'];x=matrices[mode];metadata=result['metadata'][mode];ids=metadata['candidates']
                        result['predictions'][name]={}
                        for seed in config['seeds']:
                            kind=method['kind']
                            if kind=='standalone':ranking=metadata['standalone'][method['ranking']]
                            else:
                                if kind=='learned':scores=models[name,seed].predict(x)
                                elif kind=='feature':scores=x[:,FEATURES.index(method['feature'])]
                                else:
                                    variant=method['variant']
                                    if variant not in fixed:fixed[variant]=FixedScorer(papers,histories,dates,q['seed'],q['date']).score(**VARIANTS[variant])
                                    scores=np.asarray([fixed[variant].get(a,0.) for a in ids])
                                ranking=[ids[j] for j in sorted(range(len(ids)),key=lambda j:(-scores[j],ids[j]))]
                            result['predictions'][name][str(seed)]=prediction(q,metadata,ranking)
                    atomic_state(path,result);artifacts.append(path)
                    saved={p.name:digest(p) for p in artifacts};atomic_state(receipt,saved)
                receipts[str(receipt.relative_to(out))]=digest(receipt)
                receipts.update({str((folder/name).relative_to(out)):sha for name,sha in saved.items()})
                intern_prediction_rows(result)
                for name in predictions:
                    for seed in config['seeds']:predictions[name][seed].append(result['predictions'][name][str(seed)])
                atomic_state(out/'progress.json',dict(**identity,completed_queries=i+1,total_queries=len(queries),worker_pid=os.getpid()))
        # Ranking receipts are complete. Release corpus/index/model state before
        # independent verification creates its own temporary ranking dictionaries.
        engines.clear();models.clear();indices=None
        del papers,histories,dates
        # The loop variable otherwise retains the last QueryEngine and its corpus.
        if 'engine' in locals():del engine
        gc.collect()
        scores={};verification={};inventory=dict(receipts)
        for name,seed_rows in predictions.items():
            scores[name]={}
            for seed,rows in seed_rows.items():
                checked=verify(rows);scores[name][str(seed)]=checked['aggregate'];verification[f'{name}/s{seed}']=checked
                path=out/f'{name}.s{seed}.predictions.json';atomic_state(path,rows);inventory[path.name]=digest(path)
        comparisons={}
        for primary,baseline in config['primary_comparisons']:
            paired=seed_average_pairs(predictions[primary],predictions[baseline],config['primary_metric'])
            name=primary+'__minus__'+baseline
            if name in comparisons:raise ValueError('Duplicate primary contrast')
            comparisons[name]=dict(**cluster_inference(paired['differences'],paired['authors'],**config['inference']),per_seed_deltas=paired['per_seed_deltas'])
        adjusted=holm({name:r['p_two_sided_cluster_signflip'] for name,r in comparisons.items()})
        for name,row in comparisons.items():row['holm_p']=adjusted[name]
        subgroups={}
        for primary,baseline in config['subgroup_comparisons']:
            name=primary+'__minus__'+baseline
            if name in subgroups:raise ValueError('Duplicate subgroup comparison')
            subgroups[name]=subgroup_compare(predictions[primary],predictions[baseline],**config['inference'])
        atomic_state(out/'subgroups.json',subgroups);inventory['subgroups.json']=digest(out/'subgroups.json')
        summary={name:{metric:dict(mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1))) if all(v is not None for v in values) else None
                       for metric in next(iter(seed_scores.values()))
                       for values in [[seed_scores[str(seed)][metric] for seed in config['seeds']]]} for name,seed_scores in scores.items()}
        atomic_state(out/'independent_metric_verification.json',dict(status='PASS',results=verification))
        for filename in ['identity.json','data_audit.json','independent_metric_verification.json']:inventory[filename]=digest(out/filename)
        report=dict(status='Completed frozen PreScience evaluation',**identity,config=config,queries=len(queries),per_seed=scores,
                    seed_summary=summary,primary_comparisons=comparisons,output_sha256=inventory,
                    limitations=['Observed coauthorship is a proxy; unobserved pairs are not verified negatives.',
                                 'Author-cluster inference omits crossed network dependence; intervals are marginal.',
                                 'No model/configuration selection performed in this runner; final models come from development.',
                                 'Full-pool factorial and end-to-end no-content modes must be interpreted separately.'])
        atomic_state(out/'report.json',report);return report
    finally:writer.unlink()


if __name__=='__main__':
    import psutil
    parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True,type=Path);parser.add_argument('--run-id',required=True)
    parser.add_argument('--ledger',required=True,type=Path);args=parser.parse_args()
    if psutil.virtual_memory().available<3500*1024**2:raise RuntimeError('Wait for3500MiB availableRAM before final inference')
    print(json.dumps(run(ROOT,args.lock,args.run_id,args.ledger),indent=2))
