"""Frozen ogbl-collab inference with tree/fixed/NCN comparators, no training.

No dataset loader is used. Only a globally locked explicit split file may be
opened, after model/code/data bindings are checked. Real test execution pending.
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
from final_evaluation_gate import authorize_evaluation,local_file,canonical
from final_statistics import cluster_inference,holm
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3]
CODE=['ogb_final_run.py','ogb_validation.py','ogb_multiseed.py','final_evaluation_gate.py','final_statistics.py','supervisor_state.py']


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def prepare(root,config):
    roles={role:{local_file(root,e['path']) for e in config['artifacts'] if e['role']==role} for role in ['code','model','data']}
    for name in CODE:
        if Path(__file__).with_name(name).resolve() not in roles['code']:raise ValueError('Unlocked runner dependency: '+name)
    if config.get('runner_schema')!=1 or config.get('graph_cutoff_year')!=2017:raise ValueError('Only frozen training-graph context through2017 is implemented')
    inputs=config.get('inputs',{})
    if set(inputs)!={'edges','years','weights','nodes','split'}:raise ValueError('Explicit locked graph/node/split inputs required')
    paths={name:local_file(root,path) for name,path in inputs.items()}
    if not set(paths.values())<=roles['data']:raise ValueError('All OGB input files must be locked data')
    if type(config.get('negative_count')) is not int or config['negative_count']<50:raise ValueError('Freeze shared negative count')
    if type(config.get('batch_size')) is not int or config['batch_size']<1:raise ValueError('Freeze inference batch size')
    if set(config.get('inference',{}))!={'repetitions','seed'}:raise ValueError('Freeze paired inference settings')
    if type(config['inference']['repetitions']) is not int or config['inference']['repetitions']<100 or type(config['inference']['seed']) is not int:
        raise ValueError('Invalid paired inference settings')
    models={};needs_ncn=False
    for name,method in config['methods'].items():
        if not re.fullmatch(r'[A-Za-z0-9_+.\-]+',name):raise ValueError('Invalid method name')
        if method.get('kind') in ['tree','ncn']:
            if set(method['models'])!=set(map(str,config['seeds'])):raise ValueError('Missing model seeds')
            for seed in config['seeds']:
                path=local_file(root,method['models'][str(seed)])
                if path not in roles['model']:raise ValueError('Model not globally locked')
                models[name,seed]=path
            if method['kind']=='tree' and method.get('track') not in ['graph','graph_plus_released_nodes']:raise ValueError('Unknown tree feature track')
            needs_ncn |= method['kind']=='ncn'
        elif method.get('kind')=='feature':
            if method.get('feature') not in ['common_neighbors','adamic_adar','resource_allocation','jaccard','log_degree_product','log_degree_min','log_degree_max','prior_weight','released_node_cosine']:
                raise ValueError('Unknown fixed graph feature')
        else:raise ValueError('Unknown OGB inference kind')
    if needs_ncn:
        if Path(__file__).with_name('ncn_temporal.py').resolve() not in roles['code']:raise ValueError('NCN architecture adapter not locked')
        for name in ['model.py','utils.py']:
            if (root/'work/NeuralCommonNeighbor'/name).resolve() not in roles['code']:raise ValueError('Upstream NCN sources not locked')
        if 'torch' in sys.modules and not Path(sys.modules['torch'].__file__).resolve().is_relative_to(root/'work/ncn-runtime'):
            raise RuntimeError('NCN requires a fresh process with isolated Torch runtime')
        sys.path.insert(0,str(root/'work/ncn-runtime'));sys.path.insert(1,str(root/'work/NeuralCommonNeighbor'))
    return paths,models,needs_ncn


def success(predictions,positive_count,negative_count,evaluator):
    values=np.asarray(predictions).reshape(-1)
    if len(values)!=positive_count+negative_count or not np.isfinite(values).all():raise ValueError('Invalid final score vector')
    positive,negative=values[:positive_count],values[positive_count:]
    passed=(positive>np.partition(negative,-50)[-50]).astype(float)
    official=float(evaluator.eval(dict(y_pred_pos=positive,y_pred_neg=negative))['hits@50'])
    if abs(official-float(passed.mean()))>1e-12:raise ValueError('Official/independent Hits50 mismatch')
    return passed,official


def ncn_scores(torch,checkpoint,nodes,edges,pairs,batch_size):
    from ncn_temporal import make_models
    from torch_sparse import SparseTensor
    model,predictor=make_models(torch)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);predictor.load_state_dict(saved['predictor']);model.eval();predictor.eval()
    adj=SparseTensor.from_edge_index(torch.from_numpy(edges.T.copy()),sparse_sizes=(len(nodes),len(nodes))).to_symmetric().coalesce()
    query=torch.from_numpy(pairs.T.copy());chunks=[]
    with torch.inference_mode():
        embedding=model(torch.from_numpy(nodes),adj)
        for start in range(0,len(pairs),batch_size):chunks.append(predictor(embedding,adj,query[:,start:start+batch_size]).flatten().numpy())
    return np.concatenate(chunks)


def run(root,lock_path,run_id,ledger):
    root=Path(root).resolve();config=authorize_evaluation(root,lock_path,'ogbl-collab',run_id,ledger)
    paths,models,needs_ncn=prepare(root,config)
    relative=Path(config['output_directory']);out=(root/relative).resolve()
    if relative.is_absolute() or not out.is_relative_to(root):raise ValueError('Output must stay in workspace')
    identity=dict(run_id=run_id,lock_sha256=read(lock_path)['lock_sha256'],config_sha256=hashlib.sha256(canonical(config)).hexdigest())
    out.mkdir(parents=True,exist_ok=True)
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Different final run identity')
    writer=out/'writer.lock';fd=os.open(writer,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid()),stream)
    try:
        atomic_state(out/'identity.json',identity)
        if (out/'report.json').exists():
            report=read(out/'report.json')
            for name,sha in report['output_sha256'].items():
                if digest(out/name)!=sha:raise ValueError('Completed output changed')
            return report
        import torch
        import pandas as pd
        from lightgbm import Booster
        from ogb.linkproppred import Evaluator
        from ogb_validation import features,graph,NAMES
        from ogb_multiseed import node_features
        torch.set_num_threads(1)
        if needs_ncn and (torch.__version__!='2.5.1+cpu' or not Path(torch.__file__).resolve().is_relative_to(root/'work/ncn-runtime')):
            raise RuntimeError('Wrong isolated NCN runtime')
        edges=pd.read_csv(paths['edges'],header=None).to_numpy(dtype=np.int64)
        years=pd.read_csv(paths['years'],header=None).to_numpy().ravel()
        weights=pd.read_csv(paths['weights'],header=None).to_numpy(dtype=np.float32).ravel()
        nodes=pd.read_csv(paths['nodes'],header=None,dtype=np.float32).to_numpy()
        if len(edges)!=len(years) or len(edges)!=len(weights) or years.max()>2017 or nodes.shape[1]!=128 or not np.isfinite(nodes).all():
            raise ValueError('Invalid frozen training graph/node inputs')
        if edges.min()<0 or edges.max()>=len(nodes) or not np.isfinite(weights).all():raise ValueError('Invalid graph values')
        # First label-bearing read: explicit globally locked split, never a dataset loader.
        split=torch.load(paths['split'],map_location='cpu',weights_only=False)
        positive=np.asarray(split['edge'],dtype=np.int64);negative=np.asarray(split['edge_neg'],dtype=np.int64)
        if positive.ndim!=2 or positive.shape[1]!=2 or len(positive)==0 or negative.shape!=(config['negative_count'],2):raise ValueError('Unexpected final split dimensions')
        pairs=np.concatenate([positive,negative]);del split
        if pairs.min()<0 or pairs.max()>=len(nodes):raise ValueError('Unknown node in final split')
        gx=features(pairs,graph(edges,weights,len(nodes)));nx=None
        if any(m.get('track')=='graph_plus_released_nodes' or m.get('feature')=='released_node_cosine' for m in config['methods'].values()):nx=node_features(pairs,nodes)
        evaluator=Evaluator(name='ogbl-collab');scores={};bits={};inventory={'identity.json':digest(out/'identity.json')}
        for name,method in config['methods'].items():
            scores[name]={};bits[name]={}
            for seed in config['seeds']:
                path=out/f'{name}.s{seed}.predictions.npz';receipt=path.with_suffix('.receipt')
                if receipt.exists():
                    saved=read(receipt)
                    if any(saved.get(key)!=value for key,value in identity.items()):raise ValueError('Committed prediction identity changed')
                    if saved['sha256']!=digest(path):raise ValueError('Committed prediction file changed')
                    with np.load(path,allow_pickle=False) as cache:values=np.concatenate([cache['positive'],cache['negative']])
                else:
                    if method['kind']=='feature':values=nx[:,0] if method['feature']=='released_node_cosine' else gx[:,NAMES.index(method['feature'])]
                    elif method['kind']=='tree':
                        x=gx if method['track']=='graph' else np.column_stack([gx,nx])
                        model=Booster(model_file=str(models[name,seed]))
                        if model.num_feature()!=x.shape[1]:raise ValueError('Frozen tree feature width mismatch')
                        values=model.predict(x,num_threads=1);del model,x
                    else:values=ncn_scores(torch,models[name,seed],nodes,edges,pairs,config['batch_size'])
                    passed,value=success(values,len(positive),len(negative),evaluator)
                    temporary=path.with_suffix('.part')
                    with temporary.open('wb') as stream:np.savez_compressed(stream,positive=values[:len(positive)],negative=values[len(positive):])
                    temporary.replace(path);atomic_state(receipt,dict(sha256=digest(path),**identity))
                bits[name][seed],scores[name][str(seed)]=success(values,len(positive),len(negative),evaluator)
                inventory[path.name]=digest(path);inventory[receipt.name]=digest(receipt)
                atomic_state(out/'progress.json',dict(method=name,seed=seed,worker_pid=os.getpid(),**identity));gc.collect()
        comparisons={};source=positive[:,0].astype(str)
        for primary,baseline in config['primary_comparisons']:
            name=primary+'__minus__'+baseline
            if name in comparisons:raise ValueError('Duplicate primary comparison')
            diffs=np.stack([bits[primary][seed]-bits[baseline][seed] for seed in config['seeds']])
            comparisons[name]=dict(**cluster_inference(diffs.mean(0),source,**config['inference']),
                                  per_seed_deltas={str(seed):float(row.mean()) for seed,row in zip(config['seeds'],diffs)})
        adjusted=holm({name:row['p_two_sided_cluster_signflip'] for name,row in comparisons.items()})
        for name,row in comparisons.items():row['holm_p']=adjusted[name]
        report=dict(status='Completed frozen ogbl-collab evaluation',**identity,config=config,positive_count=len(positive),negative_count=len(negative),
                    independent_metric_verification='All saved files agree with official Evaluator strict Hits50',per_seed=scores,
                    seed_summary={name:dict(mean=float(np.mean(list(values.values()))),sample_sd=float(np.std(list(values.values()),ddof=1))) for name,values in scores.items()},
                    primary_comparisons=comparisons,output_sha256=inventory,
                    limitations=['Fixed training graph through2017; no2018 validation edges added.',
                                 'NCN best of100validation checkpoints versus3tree configs; unequal selection budgets.',
                                 'Released node vectors have unverified temporal provenance.',
                                 'Source-cluster inference conditions on the fixed negative pool and omits crossed dependence.',
                                 'Repeated fixed-baseline scores over seeds do not add independent observations.'])
        atomic_state(out/'report.json',report);return report
    finally:writer.unlink()


if __name__=='__main__':
    import psutil
    parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True,type=Path);parser.add_argument('--run-id',required=True)
    parser.add_argument('--ledger',required=True,type=Path);args=parser.parse_args()
    if psutil.virtual_memory().available<3500*1024**2:raise RuntimeError('Wait for3500MiB availableRAM')
    print(json.dumps(run(ROOT,args.lock,args.run_id,args.ledger),indent=2))
