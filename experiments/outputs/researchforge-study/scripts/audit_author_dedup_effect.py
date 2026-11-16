"""Compare preserved raw-author and cleaned-author development checkpoints."""
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]
OLD=OUT/'results/text-graph-200-100-1000-s42'
NEW=OUT/'results/text-graph-dedup-200-100-1000-s42'

def main():
    features=json.loads((NEW/'report.json').read_text())['features'];result={}
    for part in ['train_queries','valid_queries']:
        old_files={p.name:p for p in (OLD/part).glob('*.json')}
        new_files={p.name:p for p in (NEW/part).glob('*.json')}
        assert old_files.keys()==new_files.keys(),'Query sample changed'
        changed=np.zeros(len(features),dtype=int);maximum=np.zeros(len(features))
        changed_pools=0;changed_queries=0;compared_rows=0
        for name,path in old_files.items():
            a=json.loads(path.read_text());b=json.loads(new_files[name].read_text())
            assert a['qid']==b['qid'] and a['seed']==b['seed'] and a['truth']==b['truth']
            changed_pools+=set(a['candidates'])!=set(b['candidates'])
            ia={aid:i for i,aid in enumerate(a['candidates'])};ib={aid:i for i,aid in enumerate(b['candidates'])}
            common=sorted(ia.keys()&ib.keys())
            with np.load(path.with_suffix('.npz')) as x, np.load(new_files[name].with_suffix('.npz')) as y:
                delta=np.abs(x['X'][[ia[c] for c in common]]-y['X'][[ib[c] for c in common]])
            mask=delta>1e-7;changed+=mask.sum(0);maximum=np.maximum(maximum,delta.max(0))
            changed_queries+=bool(mask.any());compared_rows+=len(common)
        result[part]=dict(queries=len(old_files),identical_query_ids_seed_and_truth=True,
            changed_candidate_sets=int(changed_pools),queries_with_changed_common_candidate_features=changed_queries,
            compared_common_candidate_rows=compared_rows,
            changed_features={feature:dict(rows=int(changed[i]),max_absolute_change=float(maximum[i]))
                              for i,feature in enumerate(features) if changed[i]})
    result['note']='Only common candidate rows compared for features; raw results remain preserved.'
    (OUT/'data-audit/author_dedup_effect.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
