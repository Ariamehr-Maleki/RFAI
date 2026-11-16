"""Audit pinned official frequency helper on raw and cleaned development data.

Execute only AST-extracted pure helpers, never the upstream test-default main.
The shared pool is unchanged. No exact end-to-end padding/reproduction claim.
"""
import argparse
import ast
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from prescience_pilot import ROOT,DATA,day,evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');run=Path(ap.parse_args().run)
    assert json.loads((run/'report.json').read_text())['status'].startswith('Development')
    out=run/'official-frequency-audit';out.mkdir(exist_ok=True)
    meta=[json.loads(p.read_text()) for p in sorted((run/'valid_queries').glob('*.json'))]
    authors={q['seed'] for q in meta};histories={}
    with (DATA/'author_publications.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['key'] in authors:histories[r['key']]=r['value'] or []
    required={str(pid) for ids in histories.values() for pid in ids};raw={}
    for batch in pq.ParquetFile(DATA/'train.parquet').iter_batches(batch_size=2048,columns=['corpus_id','date','authors']):
        for p in batch.to_pylist():
            if str(p['corpus_id']) in required:
                raw[str(p['corpus_id'])]=dict(date=day(p['date']),authors=[dict(author_id=str(a['author_id'])) for a in p['authors'] or [] if a.get('author_id')])
    clean={pid:dict(date=p['date'],authors=[dict(author_id=a) for a in dict.fromkeys(x['author_id'] for x in p['authors'])]) for pid,p in raw.items()}
    histories={a:[str(pid) for pid in ids if str(pid) in raw] for a,ids in histories.items()}
    assert all([raw[p]['date'] for p in ids]==sorted(raw[p]['date'] for p in ids) for ids in histories.values()),'Official histories not chronological; stop parity audit'
    sorted_hist={a:sorted(set(ids),key=lambda p:(raw[p]['date'],p)) for a,ids in histories.items()}
    sources=[(ROOT/'work/prescience/task_coauthor_prediction/dataset.py','get_preexisting_publications_for_author'),
             (ROOT/'work/prescience/task_coauthor_prediction/baseline_frequency.py','get_collaboration_frequencies')]
    env=dict(bisect_left=bisect_left);hashes={}
    for path,name in sources:
        tree=ast.parse(path.read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),env)
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    settings={'official_raw_history_raw_authors':(raw,histories),'official_raw_history_unique_authors':(clean,histories),
              'official_helper_clean_history_unique_authors':(clean,sorted_hist)}
    scores={name:[] for name in settings};changed_counts=0;changed_history=0;max_error=0.
    for q in meta:
        values={}
        for name,(papers,hist) in settings.items():
            values[name]=env['get_collaboration_frequencies'](q['seed'],q['date'],10,hist,papers)
            scores[name].extend(values[name].get(a,0) for a in q['candidates'])
        names=list(settings)
        changed_counts+=values[names[0]]!=values[names[1]]
        changed_history+=values[names[1]]!=values[names[2]]
        with np.load(run/'valid_queries'/(q['qid']+'.npz')) as z:
            expected=z['X'][:,0]
        actual=np.array([values[names[2]].get(a,0) for a in q['candidates']])
        max_error=max(max_error,float(np.max(np.abs(expected-actual))))
        np.testing.assert_array_equal(actual,expected)
    groups=np.array([len(q['candidates']) for q in meta]);metrics={}
    for name,values in scores.items():
        metrics[name],rows=evaluate(np.array(values),groups,meta,True)
        (out/f'{name}.predictions.json').write_text(json.dumps(rows))
    report=dict(status='Pinned official pure-helper parity on100 DEVELOPMENT shared candidate pools; not full official run',
                queries=len(meta),count_changes_from_author_dedup=changed_counts,count_changes_from_history_policy=changed_history,
                clean_feature_max_error=max_error,metrics=metrics,official_source_hashes=hashes,
                limitations=['Shared pool and author-ID tie policy are study adaptations.',
                             'Upstream random padding over all author keys and iterated oracle target-count option are not executed.',
                             'Only nonempty author IDs considered; all official test labels sealed.'])
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
