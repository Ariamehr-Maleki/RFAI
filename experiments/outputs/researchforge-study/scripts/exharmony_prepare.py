"""Audit corpus/query separation and freeze an adapted author-disjoint query split.

Uses query metadata for grouping, never opens relevance TSVs or computes scores.
The released benchmark has only test queries; this adaptation must be labeled.
"""
import collections
import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'work/data/exHarmony'
OUT=ROOT/'outputs/researchforge-study/data-audit'

def norm(text):
    return re.sub(r'\W+',' ',(text or '').casefold()).strip()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    queries={};duplicates=0
    with (DATA/'papers_test.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            duplicates+=r['id'] in queries
            queries[r['id']]={'title':norm(r.get('title')),'authors':[]}
    judgeable=set()
    with (DATA/'papers_test_judgeable.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);judgeable.add(r['id'])
            if r['id'] in queries:
                queries[r['id']]['authors']=sorted({a['author']['id'] for a in r.get('authorships',[]) if a.get('author',{}).get('id')})
    ids=sorted(judgeable&queries.keys())
    parent={q:q for q in ids}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    first_author,first_title={},{}
    for q in ids:
        for a in queries[q]['authors']:
            if a in first_author:union(q,first_author[a])
            else:first_author[a]=q
        title=queries[q]['title']
        if title:
            if title in first_title:union(q,first_title[title])
            else:first_title[title]=q
    components=collections.defaultdict(list)
    for q in ids:components[root(q)].append(q)
    groups=sorted(components.values(),key=lambda g:(-len(g),hashlib.sha256(min(g).encode()).hexdigest()))
    split={k:[] for k in ['train','validation','test']}
    fractions={'train':.6,'validation':.2,'test':.2}
    # Two query/qrel records were displayed during format inspection; quarantine
    # their entire author-connected components into development training.
    inspected={'https://openalex.org/W4385781987','https://openalex.org/W4382568152'}
    for group in groups:
        target='train' if inspected.intersection(group) else max(split,key=lambda k:fractions[k]*len(ids)-len(split[k]))
        split[target].extend(group)
    for k in split:split[k].sort()
    author_sets={k:{a for q in v for a in queries[q]['authors']} for k,v in split.items()}
    assert not author_sets['train']&author_sets['validation']
    assert not author_sets['train']&author_sets['test']
    assert not author_sets['test']&author_sets['validation']
    title_set={q['title'] for q in queries.values() if q['title']}
    query_set=set(queries)
    excluded=[];n=0;missing=0;id_overlap=0;title_overlap=0
    collection_ids=set();duplicate_collection=0
    with gzip.open(DATA/'papers_collection.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);n+=1
            duplicate_collection+=r['id'] in collection_ids
            collection_ids.add(r['id'])
            missing+=not ((r.get('title') or '').strip() or (r.get('abstract') or '').strip())
            id_match=r['id'] in query_set
            title_match=norm(r.get('title')) in title_set
            id_overlap+=id_match;title_overlap+=title_match
            if id_match or title_match:excluded.append(r['id'])
            if n%200000==0:print(f'Audited {n} corpus records',flush=True)
    author_records=0;empty_works=0;missing_mappings=0;links_to_excluded=0
    excluded_set=set(excluded)
    with gzip.open(DATA/'authors_works_collection_ids.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);author_records+=1
            empty_works+=not r.get('works')
            for p in r.get('works',[]):
                missing_mappings+=p not in collection_ids
                links_to_excluded+=p in excluded_set
    report=dict(status='Data preparation only; no test relevance TSVs read or model evaluation',
        total_query_ids=len(queries),duplicate_query_ids=duplicates,judgeable_ids=len(judgeable),
        eligible_judgeable_query_ids=len(ids),missing_authorships=sum(not queries[q]['authors'] for q in ids),
        components=len(groups),largest_component_sizes=[len(g) for g in groups[:20]],
        split_counts={k:len(v) for k,v in split.items()},split_policy='60/20/20 target sizes; greedy whole author-connected components; duplicate normalized titles linked; inspected examples quarantined to train',
        corpus_rows=n,duplicate_corpus_ids=duplicate_collection,missing_text=missing,
        query_id_overlap=id_overlap,query_title_overlap=title_overlap,excluded_corpus_documents=len(excluded),
        author_mapping_records=author_records,empty_author_histories=empty_works,
        missing_paper_mapping_links=missing_mappings,mapping_links_to_excluded_documents=links_to_excluded,
        limitations=['Authors disjoint means query authors; citation-based relevant authors may overlap.',
            'Corpus records have no dates, so original temporal construction cannot be independently rechecked.',
            'Current citations/works_count/experience fields will not be used; no timestamp provenance.',
            'This is a new held-out adaptation of an official test-only benchmark, not its published protocol.'])
    if min(map(len,split.values()))<500:
        report['split_status']='Rejected: inadequate held-out size; inspect component distribution before use'
    else:
        report['split_status']='Frozen before model training and relevance inspection'
        target=OUT/'exharmony_split.json'
        if target.exists():assert json.loads(target.read_text())==split,'Existing split differs; do not overwrite'
        else:target.write_text(json.dumps(split,indent=2))
        report['split_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (OUT/'exharmony_excluded_corpus_ids.json').write_text(json.dumps(sorted(excluded)))
    (OUT/'exharmony_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
