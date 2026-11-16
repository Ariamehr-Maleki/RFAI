"""Supplementary diagnostics of saved frozen predictions; no fitting or tuning."""
import gc
import hashlib
import json
from pathlib import Path
import sys
from datetime import date,timedelta
import numpy as np
HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
sys.path.insert(0,str(STUDY/'scripts'))
from final_statistics import cluster_inference

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    pairs=[('full_lambdamart','full_binary_lightgbm'),
           ('factorial_text+network+explicit_time','factorial_network+explicit_time')]
    manifest={};data={};metadata=None
    for name in dict.fromkeys(n for pair in pairs for n in pair):
        data[name]={}
        for seed in [42,43,44]:
            p=STUDY/f'results/final/prescience/{name}.s{seed}.predictions.json'
            rows=read(p);manifest[p.name]=digest(p)
            data[name][seed]={r['qid']:r['ndcg1000'] for r in rows}
            if metadata is None:metadata={r['qid']:{k:r[k] for k in ['qid','seed','truth','date']} for r in rows}
            if set(data[name][seed])!=set(metadata):raise ValueError('Query mismatch')
            del rows;gc.collect()
    qids=sorted(metadata);parent=list(range(len(qids)));first={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,qid in enumerate(qids):
        r=metadata[qid]
        for a in set([r['seed']]+r['truth']):
            if a in first:parent[root(i)]=root(first[a])
            else:first[a]=i
    clusters=[str(root(i)) for i in range(len(qids))]
    _,counts=np.unique(clusters,return_counts=True)
    results={}
    for a,b in pairs:
        differences=np.array([np.mean([data[a][s][q]-data[b][s][q] for s in [42,43,44]]) for q in qids])
        quarters={}
        for i,q in enumerate(qids):
            d=date(1970,1,1)+timedelta(days=metadata[q]['date']);label=f'{d.year}-Q{(d.month-1)//3+1}'
            quarters.setdefault(label,[]).append(float(differences[i]))
        results[a+' minus '+b]=dict(shared_observed_team_components=cluster_inference(differences,clusters),
            quarters={k:dict(queries=len(v),delta=float(np.mean(v))) for k,v in quarters.items()})
    out=dict(status='Post-result supplementary dependence/period diagnostic; not a new primary test',
        components=len(counts),largest_component=int(max(counts)),component_sizes=sorted(counts.tolist(),reverse=True),
        results=results,input_sha256=manifest,
        limitations=['Components join queries sharing any observed target-team author, including seed.',
            'Still excludes dependence through unobserved shared candidates/historical neighbors.',
            'Clustering uses labels only for uncertainty calculation, never model inputs.',
            'Quarter means are descriptive; no model or subgroup selection from these results.'])
    (HERE/'dependence-robustness.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['component_sizes','input_sha256']},indent=2))

if __name__=='__main__':main()
