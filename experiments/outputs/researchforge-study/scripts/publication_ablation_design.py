"""Pre-test component definitions, separate from the already queued pilot.

No training or label reads occur on import. Explicit time means age/decay, not
removal of chronology or recent-publication windows used by shared profiles.
"""
from itertools import combinations

CONTROLS={'log_seed_history','log_candidate_history','history_balance'}
COMPONENTS={
    'text':{'category_cosine','lexical_mean10','lexical_max10','lexical_retrieval',
            'semantic_mean10','semantic_max10','semantic_retrieval'},
    'network':{'frequency10','frequency45','frequency_all','common_neighbors10','jaccard10',
               'log_degree_product','log_candidate_degree','reverse_frequency10','citation_pull'},
    'explicit_time':{'recency_collaboration','activity_decay'},
}
# These scores mix evidence; omit from EVERY factorial cell, including full.
COMPOSITES={'autoresearch_description','rrf'}

def factorial_columns(feature_names):
    names=list(feature_names)
    if len(names)!=len(set(names)):raise ValueError('Duplicate feature name')
    classified=CONTROLS|COMPOSITES|set().union(*COMPONENTS.values())
    if set(names)!=classified:raise ValueError('Feature schema differs; review component mapping')
    result={}
    for size in range(4):
        for enabled in combinations(COMPONENTS,size):
            keep=CONTROLS|set().union(*(COMPONENTS[k] for k in enabled))
            label='+'.join(enabled) if enabled else 'controls_only'
            result[label]=[i for i,f in enumerate(names) if f in keep]
    return result

def history_group(count):
    if count<1:raise ValueError('A seed must have prior history')
    return '1-3' if count<=3 else '4-10' if count<=10 else '11+'
