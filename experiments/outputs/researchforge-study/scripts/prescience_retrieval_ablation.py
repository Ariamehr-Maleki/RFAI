"""Content-free candidate generation and features; base pilot stays unchanged."""
import numpy as np
from prescience_hybrid_graph import QueryEngine,FEATURES
from publication_ablation_design import COMPONENTS

class ZeroTextIndex:
    def __init__(self,ids):
        self.ids=list(ids);self.row={p:i for i,p in enumerate(self.ids)}
    def similarities(self,historical_ids,cutoff):return np.zeros(len(self.ids),dtype=np.float32)

def model_columns(mode):
    if mode=='full':return list(range(len(FEATURES)))
    if mode=='no_content':return [i for i,f in enumerate(FEATURES) if f not in COMPONENTS['text']]
    raise ValueError('Unknown content mode')

class NoContentEngine(QueryEngine):
    def __init__(self,papers,histories,dates):
        zero=ZeroTextIndex(sorted(papers))
        super().__init__(papers,histories,dates,zero,zero)
    def run(self,*args,**kwargs):
        x,y,metadata=super().run(*args,**kwargs)
        # Categories are content too. Zero as defense in depth and exclude from fit.
        for i,f in enumerate(FEATURES):
            if f in COMPONENTS['text']:x[:,i]=0
        metadata['content_policy']='No lexical/semantic candidate channels; all content features zero and excluded from fit'
        for name in ['lexical_standalone','semantic_standalone']:
            metadata['standalone'].pop(name,None)
        return x,y,metadata
