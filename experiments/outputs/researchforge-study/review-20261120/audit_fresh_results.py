"""Independent metric/inference check after all fresh-cohort stages complete."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from check_inference_families import infer,adjust
R=Path(__file__).resolve().parent;O=R/'fresh-cohort'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metrics(pred,truth,new):
    truth=set(truth);new=set(new);repeat=truth-new;top=set(pred[:10]);hits=[i+1 for i,a in enumerate(pred) if a in truth]
    def ndcg(k):return math.fsum(1/math.log2(j+1) for j in hits if j<=k)/math.fsum(1/math.log2(j+1) for j in range(1,min(k,len(truth))+1))
    return {'ndcg10':ndcg(10),'ndcg1000':ndcg(1000),'rprecision':sum(j<=len(truth) for j in hits)/len(truth),'recall10':len(top&truth)/len(truth),'recall50':len(set(pred[:50])&truth)/len(truth),'new_recall10':len(top&new)/len(new) if new else None,'repeat_recall10':len(top&repeat)/len(repeat) if repeat else None,'candidate_recall':len(set(pred)&truth)/len(truth)}
def main():
    manifest=read(O/'sample-manifest.json');identity=sha(O/'sample-manifest.json');expected=read(O/'results.json')
    assert expected['sample_sha256']==identity
    rows={m:[] for m in ['ranker','classifier','official']};groups=[];largest=0.
    for q in manifest['queries']:
        r=read(O/'ranker-queries'/(q['qid']+'.json'));p=O/'official-queries'/(q['qid']+'.json');o=read(p)
        assert sha(p)==read(p.with_suffix('.receipt.json'))['sha256'] and o['sample_sha256']==identity
        assert all(o[k]==q[k] for k in q) and o['truth']==r['query']['truth']
        assert o['new_truth']==r['metadata']['new_truth'] and len(o['predicted'])==len(set(o['predicted']))==1000
        assert q['seed'] not in o['predicted'];groups.append(q['seed'])
        actual=metrics(o['predicted'],o['truth'],o['new_truth'])
        for k,v in actual.items():
            if v is None:assert o[k] is None
            else:largest=max(largest,abs(v-o[k]));assert abs(v-o[k])<1e-12
        rows['official'].append(actual)
        for name,method in [('ranker','full_lambdamart'),('classifier','full_binary_lightgbm')]:
            seed_values=[metrics(r['predictions'][method+'.s'+str(seed)]['predicted'],r['query']['truth'],r['metadata']['new_truth']) for seed in [42,43,44]]
            rows[name].append({k:float(np.mean([s[k] for s in seed_values])) if seed_values[0][k] is not None else None for k in seed_values[0]})
    summaries={name:{k:float(np.mean([x[k] for x in rr if x[k] is not None])) for k in rr[0]} for name,rr in rows.items()}
    for name,values in expected['summary'].items():
        for k,v in values.items():largest=max(largest,abs(v-summaries[name][k]));assert abs(v-summaries[name][k])<1e-12
    family={}
    for other in ['official','classifier']:
        delta=np.array([a['ndcg1000']-b['ndcg1000'] for a,b in zip(rows['ranker'],rows[other])]);family['ranker_minus_'+other]=infer(delta,groups)
    adjust(family)
    for k,v in family.items():
        ref=expected['contrasts'][k]
        errors=[abs(v['delta']-ref['delta']),max(abs(x-y) for x,y in zip(v['ci95'],ref['ci95'])),abs(v['holm_p']-ref['holm_p_two_contrasts'])]
        assert max(errors)<1e-12,(k,errors);largest=max(largest,max(errors))
    result={'status':'PASS','queries':len(groups),'clusters':len(set(groups)),'sample_sha256':identity,'summary':summaries,'contrasts':family,'max_numeric_difference':largest,'script_sha256':sha(Path(__file__)),'scope':'All official output metrics, all summary metrics and both prespecified seed-author clustered contrasts. Does not independently reimplement the official scoring algorithm.'}
    (R/'fresh-independent-results-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='summary'}),flush=True)
if __name__=='__main__':main()
