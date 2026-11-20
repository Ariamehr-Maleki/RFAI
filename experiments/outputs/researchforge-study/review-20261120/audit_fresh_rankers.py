"""Independent full cached-model replay and metrics; no original scorer import."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from lightgbm import Booster
R=Path(__file__).resolve().parent;S=R.parent;ROOT=S.parents[1];O=R/'fresh-cohort'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def main():
    manifest=read(O/'sample-manifest.json');identity=sha(O/'sample-manifest.json')
    cfg=read(S/'final/global-lock.json')['payload']['plan']['datasets']['prescience'];models={}
    for name in ['full_lambdamart','full_binary_lightgbm']:
        for seed,spec in cfg['methods'][name]['models'].items():
            p=ROOT/spec['path'];assert sha(p)==spec['sha256']
            models[name+'.s'+seed]=(Booster(model_file=str(p)),[cfg['feature_names'].index(x) for x in spec['features']])
    count=0;max_error=0.;inventory=[]
    for i,q in enumerate(manifest['queries']):
        p=O/'ranker-queries'/(q['qid']+'.json');r=read(p)
        assert r['sample_sha256']==identity and sha(p)==read(p.with_suffix('.receipt.json'))['sha256']
        assert sha(p.with_suffix('.npz'))==r['cache_sha256']
        assert all(r['query'][k]==q[k] for k in q)
        ids=r['metadata']['candidates'];truth=set(r['query']['truth']);new=set(r['metadata']['new_truth']);repeat=truth-new
        assert len(ids)==1000 and len(set(ids))==1000 and r['query']['seed'] not in ids
        assert abs(len(set(ids)&truth)/len(truth)-r['metadata']['candidate_recall'])<1e-12
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as c:
            X=c['X'];assert np.isfinite(X).all();assert np.array_equal(c['y'],[int(x in truth) for x in ids])
            for name,(model,cols) in models.items():
                score=model.predict(X[:,cols],num_threads=1)
                pred=[ids[j] for j in sorted(range(len(ids)),key=lambda j:(-score[j],ids[j]))]
                old=r['predictions'][name];assert pred==old['predicted'],(q['qid'],name)
                hit=[j+1 for j,a in enumerate(pred) if a in truth]
                metrics={'ndcg1000':math.fsum(1/math.log2(j+1) for j in hit)/math.fsum(1/math.log2(j+1) for j in range(1,min(1000,len(truth))+1)),
                         'ndcg10':math.fsum(1/math.log2(j+1) for j in hit if j<=10)/math.fsum(1/math.log2(j+1) for j in range(1,min(10,len(truth))+1)),
                         'rprecision':sum(j<=len(truth) for j in hit)/len(truth),
                         'recall10':len(set(pred[:10])&truth)/len(truth),
                         'recall50':len(set(pred[:50])&truth)/len(truth),
                         'new_recall10':len(set(pred[:10])&new)/len(new) if new else None,
                         'repeat_recall10':len(set(pred[:10])&repeat)/len(repeat) if repeat else None}
                for k,v in metrics.items():
                    if v is None:assert old[k] is None
                    else:max_error=max(max_error,abs(v-old[k]));assert abs(v-old[k])<1e-12,(q['qid'],name,k)
                count+=1
        inventory.append({'qid':q['qid'],'json_sha256':sha(p),'npz_sha256':sha(p.with_suffix('.npz'))})
        if (i+1)%100==0:print('independent replay',i+1,flush=True)
    result={'status':'PASS','queries':len(inventory),'rankings':count,'max_metric_error':max_error,'sample_sha256':identity,'script_sha256':sha(Path(__file__)),'scope':'All six saved rankings per query, nDCG10/1000, R-precision, recall10/50, new/repeat recall10, candidate recall; no raw-feature reconstruction in this check','files':inventory}
    (R/'fresh-independent-ranker-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
