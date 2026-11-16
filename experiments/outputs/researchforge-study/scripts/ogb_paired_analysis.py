"""Exploratory source-cluster comparisons of saved DEVELOPMENT predictions."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'outputs/researchforge-study/results/ogb-expanded-10000'

def hits(path):
    with np.load(path) as data:
        threshold=np.partition(data['negative'],-50)[-50]
        return (data['positive']>threshold).astype(float)

def main():
    report=json.loads((RUN/'report.json').read_text())
    valid=torch.load(ROOT/'work/data/ogbl-collab/extracted/collab/split/time/valid.pt',weights_only=False,map_location='cpu')
    _,inverse=np.unique(np.asarray(valid['edge'])[:,0],return_inverse=True)
    count=np.bincount(inverse);comparisons={}
    aa=hits(RUN/'adamic_adar.predictions.npz')
    for seed in report['config']['seeds']:
        graph=hits(RUN/f's{seed}-graph__lambdamart.predictions.npz')
        classifier=hits(RUN/f's{seed}-graph__binary_lightgbm.predictions.npz')
        node=hits(RUN/f's{seed}-graph_plus_released_nodes__lambdamart.predictions.npz')
        comparisons[f's{seed}:ltr_minus_classifier']=graph-classifier
        comparisons[f's{seed}:ltr_minus_adamic_adar']=graph-aa
        comparisons[f's{seed}:ltr_node_track_minus_graph']=node-graph
    names=list(comparisons)
    sums=np.column_stack([np.bincount(inverse,weights=comparisons[n]) for n in names])
    rng=np.random.default_rng(8863);boot=np.empty((2000,len(names)))
    for i in range(len(boot)):
        idx=rng.integers(len(count),size=len(count))
        boot[i]=sums[idx].sum(0)/count[idx].sum()
    output=dict(status='exploratory validation-selected comparisons; no confirmatory claim',
        clusters=len(count),repeats=len(boot),seed=8863,
        limitations=['Fixed shared negative pool; thresholds held fixed in bootstrap.',
                     'Source-only clusters omit crossed/shared-target network dependence.',
                     'Intervals are marginal and not multiplicity adjusted.',
                     'No final test outcomes read.'],
        comparisons={name:dict(delta=float(comparisons[name].mean()),
            ci95=np.quantile(boot[:,i],[.025,.975]).tolist()) for i,name in enumerate(names)})
    (RUN/'paired_exploratory.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':main()
