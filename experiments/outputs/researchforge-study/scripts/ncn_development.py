"""Pinned upstream NCN model classes; CPU temporal DEVELOPMENT adapter.

No upstream dataset loader/evaluation loop: those read test labels. Training
uses graph<=2016 and2017 labels; validation uses graph<=2017 and2018 labels.
This adapted protocol is not an exact reproduction of the published NCN score.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
RUNTIME=ROOT/'work/ncn-runtime'
REPO=ROOT/'work/NeuralCommonNeighbor'
sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(REPO))

def make_models(torch):
    from model import GCN, CNLinkPredictor
    # Published collab architecture/dropouts, reused from unmodified model.py.
    gcn=GCN(128,64,64,1,.1,ln=True,res=True,conv_fn='gcn',edrop=.25,
            xdropout=.25,taildropout=.05)
    predictor=CNLinkPredictor(64,64,1,3,.3,edrop=0.,ln=True,
                             cndeg=-1,use_xlin=True,tailact=True)
    return gcn,predictor

def smoke(torch):
    from torch_sparse import SparseTensor
    torch.manual_seed(42)
    gcn,predictor=make_models(torch)
    x=torch.randn(6,128)
    edge=torch.tensor([[0,1,1,2,2,3],[1,2,3,3,4,5]])
    adj=SparseTensor.from_edge_index(edge,sparse_sizes=(6,6)).to_symmetric()
    score=predictor(gcn(x,adj),adj,torch.tensor([[0,0,1],[2,5,4]])).flatten()
    assert score.shape==(3,) and torch.isfinite(score).all()
    score.square().mean().backward()
    gradients=[p.grad for p in list(gcn.parameters())+list(predictor.parameters()) if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    result=dict(status='tiny-graph forward/backward passed; not benchmark evaluation',torch=torch.__version__,
                torch_path=torch.__file__,gradient_tensors=len(gradients))
    out=ROOT/'outputs/researchforge-study/results/ncn-smoke';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--train-positives',type=int,default=1000);ap.add_argument('--epochs',type=int,default=1)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--batch-size',type=int,default=4096)
    args=ap.parse_args()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert Path(torch.__file__).resolve().is_relative_to(RUNTIME.resolve()), 'Isolated runtime required'
    if args.smoke:smoke(torch);return
    import numpy as np
    import pandas as pd
    import psutil
    from torch_sparse import SparseTensor
    from ogb.linkproppred import Evaluator
    from ogb_validation import training_edges
    start=time.perf_counter();torch.manual_seed(args.seed);np.random.seed(args.seed)
    run=ROOT/'outputs/researchforge-study/results'/f'ncn-dev-{args.train_positives}-e{args.epochs}-s{args.seed}'
    run.mkdir(parents=True,exist_ok=True)
    config=dict(**vars(args),torch=torch.__version__,threads=1,
        upstream_commit='11d597013750da17ce7468e344bec756a7af39a4',
        model_sha256=hashlib.sha256((REPO/'model.py').read_bytes()).hexdigest(),
        utils_sha256=hashlib.sha256((REPO/'utils.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        graph_cutoffs=[2016,2017],label_years=[2017,2018],negative_ratio=5,
        node_features='Released128-dimensional features; temporal provenance unverified')
    if (run/'config.json').exists():
        raise RuntimeError('Existing run: inspect results/checkpoint before restarting or choose a new run')
    (run/'config.json').write_text(json.dumps(config,indent=2))
    raw=ROOT/'work/data/ogbl-collab/extracted/collab/raw'
    edges=pd.read_csv(raw/'edge.csv.gz',header=None).to_numpy(dtype=np.int64)
    years=pd.read_csv(raw/'edge_year.csv.gz',header=None).to_numpy().ravel();assert years.max()==2017
    x=torch.from_numpy(pd.read_csv(raw/'node-feat.csv.gz',header=None,dtype=np.float32).to_numpy())
    assert x.shape[1]==128
    def adjacency(e):
        return SparseTensor.from_edge_index(torch.from_numpy(e.T.copy()),sparse_sizes=(len(x),len(x))).to_symmetric().coalesce()
    train_adj=adjacency(edges[years<=2016]);valid_adj=adjacency(edges)
    positive=edges[years==2017];rng=np.random.default_rng(args.seed)
    if len(positive)>args.train_positives:positive=positive[rng.choice(len(positive),args.train_positives,replace=False)]
    te,ty,_=training_edges(positive,edges,len(x),args.seed)
    te=torch.from_numpy(te.T.copy());ty=torch.from_numpy(ty.astype(bool))
    valid=torch.load(raw.parent/'split/time/valid.pt',map_location='cpu',weights_only=False)
    vp,vn=np.asarray(valid['edge']),np.asarray(valid['edge_neg']);assert len(vn)==100000
    ve=torch.from_numpy(np.concatenate([vp,vn]).T.copy())
    model,predictor=make_models(torch)
    optimizer=torch.optim.Adam([{'params':model.parameters(),'lr':.0082},{'params':predictor.parameters(),'lr':.0037}])
    evaluator=Evaluator(name='ogbl-collab');history=[];best=-1
    print(json.dumps(dict(stage='loaded',worker_pid=__import__('os').getpid(),rows=len(ty),rss=psutil.Process().memory_info().rss)),flush=True)
    for epoch in range(1,args.epochs+1):
        began=time.perf_counter();model.train();predictor.train();losses=[]
        order=torch.randperm(len(ty))
        for pos in range(0,len(ty),args.batch_size):
            idx=order[pos:pos+args.batch_size];labels=ty[idx]
            if not labels.any() or labels.all():raise RuntimeError('Batch lacks one class; change batching, not labels')
            optimizer.zero_grad();h=model(x,train_adj)
            scores=predictor(h,train_adj,te[:,idx]).flatten()
            loss=torch.nn.functional.softplus(-scores[labels]).mean()+torch.nn.functional.softplus(scores[~labels]).mean()
            assert torch.isfinite(loss);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        training_seconds=time.perf_counter()-began;model.eval();predictor.eval()
        with torch.inference_mode():
            h=model(x,valid_adj);chunks=[]
            for pos in range(0,ve.shape[1],args.batch_size):
                chunks.append(predictor(h,valid_adj,ve[:,pos:pos+args.batch_size]).flatten().numpy())
            pred=np.concatenate(chunks)
        assert np.isfinite(pred).all()
        score=float(evaluator.eval({'y_pred_pos':pred[:len(vp)],'y_pred_neg':pred[len(vp):]})['hits@50'])
        row=dict(epoch=epoch,loss=float(np.mean(losses)),validation_hits50=score,training_seconds=training_seconds,
                 epoch_seconds=time.perf_counter()-began,rss=psutil.Process().memory_info().rss)
        history.append(row);print(json.dumps(row),flush=True)
        if score>best:
            best=score;torch.save({'model':model.state_dict(),'predictor':predictor.state_dict(),'epoch':epoch},run/'best.pt')
            np.savez_compressed(run/'best.predictions.npz',positive=pred[:len(vp)],negative=pred[len(vp):])
        torch.save({'model':model.state_dict(),'predictor':predictor.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},run/'last.pt')
        (run/'history.json').write_text(json.dumps(history,indent=2))
    report=dict(status='CPU development/resource pilot; not published reproduction or final test',config=config,
                best_validation_hits50=best,history=history,wall_seconds=time.perf_counter()-start,
                deviations=['Temporal2016/2017 training instead of original random minibatch edge masking.',
                'Sampled positives and fixed5:1 negatives, CPU batches4096 rather than published65536.',
                'Early small pilot; architecture hyperparameters copied, training budget not matched to published100 epochs.',
                'Released node features have unverified temporal provenance.'])
    (run/'report.json').write_text(json.dumps(report,indent=2));print('Complete',flush=True)

if __name__=='__main__':main()
