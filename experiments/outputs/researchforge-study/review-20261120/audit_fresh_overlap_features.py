"""Reconstruct every potentially overlapping query from the frozen raw inputs."""
from pathlib import Path
import json,gc
import numpy as np
from threadpoolctl import threadpool_limits
import fresh_cohort as f
from fresh_io_v4 import setup
R=Path(__file__).resolve().parent
def main():
    setup();cfg=f.prelock();f.memory_gate(3000,'before-overlap-feature-audit')
    papers,histories,dates=f.load_slim(cfg)
    lexical=f.TextIndex(f.ROOT/cfg['index_inputs']['lexical'])
    semantic=f.SemanticIndex(f.ROOT/cfg['index_inputs']['semantic'],f.ROOT/cfg['index_inputs']['lexical'])
    engine=f.QueryEngine(papers,histories,dates,lexical,semantic)
    manifest=f.read(f.OUT/'sample-manifest.json');records=[]
    # 1-based 85–102 are the conservative window reported during overlap.
    # Additional positions are fixed operational checks, not score-selected.
    positions=sorted(set(range(84,102))|{0,426,999})
    with threadpool_limits(limits=1):
        for pos in positions:
            q=manifest['queries'][pos];p=f.OUT/'ranker-queries'/(q['qid']+'.json');r=f.read(p)
            X,y,m=engine.run(r['query'],1000,42,False)
            with np.load(p.with_suffix('.npz'),allow_pickle=False) as c:
                assert np.array_equal(X,c['X']) and np.array_equal(y,c['y']),q['qid']
            assert m['candidates']==r['metadata']['candidates'],q['qid']
            assert m['new_truth']==r['metadata']['new_truth'],q['qid']
            records.append({'position':pos+1,'qid':q['qid'],'features_labels_candidates_new_truth':'exact parity'})
            print('feature parity',pos+1,flush=True)
    f.write(R/'fresh-overlap-feature-audit.json',{'status':'PASS','queries':records,'sample_sha256':f.sha(f.OUT/'sample-manifest.json'),'script_sha256':f.sha(Path(__file__)),'scope':'Every query in conservative overlap interval plus three fixed positions; not reconstruction of all 1000 feature matrices'})
if __name__=='__main__':main()
