"""Explicit PreScience final-corpus adapter behind the global evaluation lock.

No data access on import and no command-line entry point. Development code is
left unchanged. This component is preparation, not a complete final runner.
"""
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq
from final_evaluation_gate import authorize_evaluation,local_file
from prescience_pilot import day,unique_author_ids,make_queries,sample_queries

COLUMNS=['corpus_id','date','roles','authors','categories','key_references']


def load_authorized_corpus(root,lock_path,run_id,ledger_dir):
    # Authorize and checksum every globally frozen artifact before ANY label read.
    config=authorize_evaluation(root,lock_path,'prescience',run_id,ledger_dir)
    source=config.get('corpus_inputs',{})
    if set(source)!={'corpus','author_history'}:raise ValueError('Explicit final corpus and author-history inputs required')
    paths={key:local_file(root,value) for key,value in source.items()}
    declared={local_file(root,a['path']) for a in config['artifacts'] if a['role']=='data'}
    if not set(paths.values())<=declared:raise ValueError('Corpus inputs must be globally locked data artifacts')
    policy=config.get('query_sampling',{})
    if set(policy)!={'count','seed'} or type(policy['count']) is not int or policy['count']<0 or type(policy['seed']) is not int:
        raise ValueError('Freeze query count (0 for census) and integer sampling seed before label access')
    papers={};seen=set();missing_date=0
    for batch in pq.ParquetFile(paths['corpus']).iter_batches(batch_size=2048,columns=COLUMNS):
        for row in batch.to_pylist():
            cid=sys.intern(str(row['corpus_id']))
            if cid in seen:raise ValueError('Duplicate corpus ID')
            seen.add(cid)
            if not row.get('date'):missing_date+=1;continue
            papers[cid]=dict(date=day(row['date']),authors=unique_author_ids(row['authors']),
                target='target' in (row['roles'] or []),categories=tuple(sys.intern(c) for c in (row['categories'] or [])),
                refs=tuple(sys.intern(str(r['corpus_id'])) for r in row['key_references'] or [] if r['corpus_id']))
    histories={};dates={};authors_seen=set()
    with paths['author_history'].open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);author=sys.intern(str(row['key']))
            if author in authors_seen:raise ValueError('Duplicate author-history key')
            authors_seen.add(author)
            ids=sorted({sys.intern(str(i)) for i in row['value'] or [] if str(i) in papers},key=lambda i:(papers[i]['date'],i))
            if ids:histories[author]=ids;dates[author]=[papers[i]['date'] for i in ids]
    eligible=make_queries(papers,histories,dates)
    if policy['count']>len(eligible):raise ValueError('Frozen query count exceeds eligible corpus; no silent resizing')
    queries=sample_queries(eligible,policy['count'],policy['seed'])
    audit=dict(corpus_rows=len(seen),papers_with_dates=len(papers),missing_dates=missing_date,
        author_histories=len(histories),eligible_targets=len(eligible),selected_queries=len(queries),
        source={k:str(p) for k,p in paths.items()},sampling=policy,
        interpretation='Target author IDs are labels; target text is absent from this adapter. QueryEngine must enforce strict date masks.')
    return config,papers,histories,dates,queries,audit
