"""Independent paired cluster inference from archived predictions and OGB IDs.
No original statistics module is imported. OGB source IDs require the pinned
provider test split because the original ZIP contains scores but not endpoints.
"""
from pathlib import Path
from zipfile import ZipFile
import hashlib,json,io,math
import numpy as np
R=Path(__file__).resolve().parent;S=R.parent;ROOT=S.parents[1]

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def ndcg(row,k):
 truth=set(row['truth']);hit=[i+1 for i,a in enumerate(row['predicted'][:k]) if a in truth]
 return math.fsum(1/math.log2(i+1) for i in hit)/math.fsum(1/math.log2(i+1) for i in range(1,min(k,len(truth))+1))

def infer(delta,groups):
 unique=sorted(set(str(x) for x in groups));index={g:i for i,g in enumerate(unique)}
 sums=np.zeros(len(unique));counts=np.zeros(len(unique),dtype=np.int64)
 for d,g in zip(delta,groups):sums[index[str(g)]]+=d;counts[index[str(g)]]+=1
 rng=np.random.default_rng(314159);boots=[];extreme=0;effect=float(np.mean(delta))
 for offset in range(0,10000,64):
  size=min(64,10000-offset);idx=rng.integers(0,len(unique),(size,len(unique)))
  boots.extend((sums[idx].sum(axis=1)/counts[idx].sum(axis=1)).tolist())
  signs=2*rng.integers(0,2,(size,len(unique)))-1
  extreme+=int(np.count_nonzero(np.abs((signs*sums).sum(axis=1)/len(delta))>=abs(effect)-1e-15))
 return {'delta':effect,'ci95':np.quantile(boots,[.025,.975]).tolist(),'p':(extreme+1)/10001,'queries':len(delta),'clusters':len(unique)}

def adjust(family):
 ordered=sorted(family,key=lambda k:family[k]['p']);last=0.
 for i,name in enumerate(ordered):
  last=max(last,min(1.,family[name]['p']*(len(ordered)-i)));family[name]['holm_p']=last

def check(z,dataset):
 base=f'outputs/researchforge-study/results/final/{dataset}/';report=json.loads(z.read(base+'report.json'))
 pairs=report['config']['primary_comparisons'];needed=set(m for pair in pairs for m in pair);data={};groups=None;qids=None
 if dataset=='ogbl-collab':
  lock=json.loads(z.read('outputs/researchforge-study/final/global-lock.json'))['payload']['artifacts']
  path=report['config']['inputs']['split'];expected=lock[path]
  derived=R/'ogb-test-source-nodes.npy';receipt=R/'ogb-cluster-provenance.json'
  if derived.exists() and receipt.exists():
   provenance=json.loads(receipt.read_text(encoding='utf-8'))
   assert provenance['provider_file']==path and provenance['provider_sha256']==expected['sha256']
   assert digest(derived)==provenance['derived_sha256']
   values=np.load(derived,allow_pickle=False);assert values.ndim==1 and len(values)==provenance['rows']==46329
   groups=[str(int(v)) for v in values]
  else:
   import torch
   actual=digest(ROOT/path);assert actual==expected['sha256']
   split=torch.load(ROOT/path,map_location='cpu',weights_only=False);groups=[str(int(v)) for v in split['edge'][:,0]]
   np.save(derived,np.asarray(groups,dtype=np.int64),allow_pickle=False)
   receipt.write_text(json.dumps({'provider_file':path,'provider_sha256':actual,'derived_file':'ogb-test-source-nodes.npy','derived_sha256':digest(derived),'rows':len(groups),'meaning':'First endpoint of each positive test edge, in released order; required for source-cluster inference; not an input to ranking.'},indent=2))
 for method in needed:
  data[method]=[]
  for seed in [42,43,44]:
   if dataset=='ogbl-collab':
    with np.load(io.BytesIO(z.read(base+f'{method}.s{seed}.predictions.npz')),allow_pickle=False) as a:
     values=(a['positive']>np.sort(a['negative'])[-50]).astype(float)
   else:
    filename=(('authors.' if dataset=='exHarmony' else '')+method+f'.s{seed}.predictions.json')
    rows=json.loads(z.read(base+filename));keyed={str(r['qid']):r for r in rows};ids=sorted(keyed)
    current_groups=[keyed[q]['seed'] for q in ids]
    if qids is None:qids=ids;groups=current_groups
    assert ids==qids and current_groups==groups
    values=np.asarray([ndcg(keyed[q],10 if dataset=='exHarmony' else 1000) for q in ids])
   data[method].append(values)
  data[method]=np.mean(data[method],axis=0)
 family={a+'__minus__'+b:infer(data[a]-data[b],groups) for a,b in pairs};adjust(family)
 for name,record in family.items():
  expected=report['primary_comparisons'][name]
  record['max_error']=max(abs(record['delta']-expected['delta']),max(abs(x-y) for x,y in zip(record['ci95'],expected['ci95'])),abs(record['holm_p']-expected['holm_p']))
  assert record['max_error']<1e-12,(dataset,name,record,expected)
 return family,(qids,groups,data)

def main():
 results={}
 with ZipFile(S/'deliverables/researchforge-reproducibility.zip') as z:
  for dataset in ['prescience','ogbl-collab','exHarmony']:
   family,state=check(z,dataset);results[dataset]=family
   if dataset=='prescience':original=state
   print('PASS',dataset,len(family),'comparisons',flush=True)
 with ZipFile(S/'deliverables/ResearchForge_Review_Extension_20261108.zip') as z:
  base='outputs/researchforge-study/review-20261107/';report=json.loads(z.read(base+'official-comparator-results.json'));qids,groups,data=original;family={}
  for track in report['results']:
   rows={r['qid']:r for r in json.loads(z.read(base+track+'.predictions.json'))};assert set(rows)==set(qids)
   family[track]=infer(data['full_lambdamart']-np.asarray([ndcg(rows[q],1000) for q in qids]),groups)
  adjust(family)
  for track,r in family.items():
   expected=report['contrasts'][track];r['max_error']=max(abs(r['delta']-expected['delta']),max(abs(x-y) for x,y in zip(r['ci95'],expected['ci95'])),abs(r['holm_p']-expected['holm_p_three_supplementary_contrasts']))
   assert r['max_error']<1e-12
  results['official-supplementary']=family
 (R/'independent-inference.json').write_text(json.dumps({'status':'PASS','families':results,'comparisons':sum(map(len,results.values())),'repetitions':10000,'seed':314159,'intervals':'Marginal 95%, cluster bootstrap; Holm separately within each declared family','script_sha256':digest(Path(__file__))},indent=2),encoding='utf-8')
 print('All 16 paired comparisons and family corrections verified',flush=True)

if __name__=='__main__':main()
