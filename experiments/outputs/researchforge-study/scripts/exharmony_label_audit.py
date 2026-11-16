"""Attainability of DEVELOPMENT qrels; held-out test relevance stays filtered."""
import json
import numpy as np
from scipy import sparse
from exharmony_pilot import OUT,INDEX,AUDIT,load_qrels

def main():
    split=json.loads((AUDIT/'exharmony_split.json').read_text())
    ids=np.load(INDEX/'author_ids.npy');counts=np.diff(sparse.load_npz(INDEX/'author_papers.npz').indptr)
    existing=np.sort(ids);eligible=np.sort(ids[counts>0])
    def count_members(values,universe):
        v=np.array(sorted(values),dtype=np.uint64);loc=np.searchsorted(universe,v)
        safe=np.minimum(loc,len(universe)-1)
        return int(np.sum((loc<len(universe))&(universe[safe]==v)))
    result={}
    for part in ['train','validation']:
        result[part]={}
        for name in ['authors','cite','simcite']:
            labels=load_qrels(name,set(split[part]));total=missing=empty=0;ceilings=[];zero_attainable=0
            for q,truth in labels.items():
                present=count_members(truth,existing);possible=count_members(truth,eligible)
                total+=len(truth);missing+=len(truth)-present;empty+=present-possible
                ceilings.append(possible/len(truth));zero_attainable+=possible==0
            result[part][name]=dict(queries=len(labels),positive_pairs=total,missing_author_ids=missing,
                pairs_with_author_emptied_by_title_filter=empty,macro_recall_ceiling=float(np.mean(ceilings)),
                queries_with_zero_attainable_relevance=zero_attainable)
    report=dict(status='Development partitions only; no test qrels parsed',results=result,
        policy='Keep original qrels/denominators; do not silently drop unreachable relevant authors.',
        note='An author absent from the released candidate mapping cannot be retrieved even by a perfect ranker.')
    (AUDIT/'exharmony_development_label_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
