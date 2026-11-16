"""Paired exploratory comparison of completed NCN and tree validation predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
from final_statistics import cluster_inference,holm
from verify_ncn_development import check

ROOT=Path(__file__).resolve().parents[3];RESULTS=ROOT/'outputs/researchforge-study/results'
SEEDS=[42,43,44]


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def successes(path,count):
    with np.load(path,allow_pickle=False) as saved:positive=saved['positive'].reshape(-1);negative=saved['negative'].reshape(-1)
    if len(positive)!=count or len(negative)!=100000 or not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError('Invalid full validation predictions')
    return (positive>np.partition(negative,-50)[-50]).astype(float)


def main():
    out=RESULTS/'ncn-expanded-10000';combined=json.loads((out/'report.json').read_text())
    audits={s:check(RESULTS/f'ncn-expanded-10000-e100-s{s}') for s in SEEDS}
    values=[audits[s]['validation_hits50'] for s in SEEDS]
    if not np.allclose(combined['per_seed'],values,rtol=0,atol=1e-12):raise ValueError('NCN combined report differs')
    # Explicit validation path only. Never invoke an OGB dataset loader/test split.
    import torch
    valid_path=ROOT/'work/data/ogbl-collab/extracted/collab/split/time/valid.pt'
    valid=torch.load(valid_path,map_location='cpu',weights_only=False)
    edge=np.asarray(valid['edge']);negative=np.asarray(valid['edge_neg'])
    if edge.shape!=(60084,2) or negative.shape!=(100000,2):raise ValueError('Unexpected OGB validation split')
    sources=edge[:,0].astype(str);artifacts={str(valid_path.relative_to(ROOT)):digest(valid_path)}
    ncn={}
    for seed in SEEDS:
        path=RESULTS/f'ncn-expanded-10000-e100-s{seed}/best.predictions.npz'
        ncn[seed]=successes(path,len(edge));artifacts[str(path.relative_to(ROOT))]=digest(path)
    tree=RESULTS/'ogb-expanded-10000';tree_report=json.loads((tree/'report.json').read_text())
    methods=['graph__lambdamart','graph__binary_lightgbm','graph_plus_released_nodes__lambdamart','adamic_adar']
    comparisons={}
    for method in methods:
        differences=[];per_seed={}
        for seed in SEEDS:
            path=tree/(('' if method=='adamic_adar' else f's{seed}-')+method+'.predictions.npz')
            baseline=successes(path,len(edge));artifacts[str(path.relative_to(ROOT))]=digest(path)
            reported=(tree_report['fixed_baselines'][method] if method=='adamic_adar' else tree_report['runs'][str(seed)]['methods'][method])['hits50']
            if abs(float(baseline.mean())-reported)>1e-12:raise ValueError('Tree prediction/report mismatch')
            diff=ncn[seed]-baseline;differences.append(diff);per_seed[str(seed)]=float(diff.mean())
        comparisons['NCN_minus_'+method]=dict(**cluster_inference(np.mean(differences,axis=0),sources,repetitions=5000),per_seed_deltas=per_seed)
    adjusted=holm({name:row['p_two_sided_cluster_signflip'] for name,row in comparisons.items()})
    for name,row in comparisons.items():row['exploratory_holm_p']=adjusted[name]
    result=dict(status='Completed DEVELOPMENT paired comparison; validation selected; no final test',
                seeds=SEEDS,ncn_mean=float(np.mean(values)),ncn_sample_sd=float(np.std(values,ddof=1)),
                per_seed_validation_hits50=values,comparisons=comparisons,artifact_sha256=artifacts,
                limitations=['NCN best of100validation checkpoints versus3tree configurations; selection budgets unequal.',
                             'NCN/released-node track uses vectors with unverified temporal provenance.',
                             'Same60084validation edges are averaged across model seeds, never tripled as independent examples.',
                             'Intervals condition on100000fixed shared negatives; source-only clustering omits crossed graph dependence.',
                             'Four exploratory contrasts and Holm do not remove adaptive validation-selection bias.',
                             'Graph-only tree comparison has less input information than NCN. No final superiority claim.'])
    (out/'paired_exploratory.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
