"""Post-result evaluation of the pinned upstream scorer; no fitting or tuning.

Run with PYTHONHASHSEED=0. Original frozen study files are read-only inputs.
"""
import argparse
import ast
from bisect import bisect_left
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, date, timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import psutil
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
from final_statistics import cluster_inference, holm

TRACKS = ['published_input', 'historical_input', 'harmonized_historical_input']
SOURCE = HERE/'upstream/task_coauthor_prediction/baseline_autoresearch.py'
DATASET = HERE/'upstream/task_coauthor_prediction/dataset.py'
EPOCH = date(1970, 1, 1)

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    part = p.with_suffix(p.suffix+'.part')
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    part.replace(p)

def official_functions():
    """Compile only original constant assignments and pure function definitions."""
    env = dict(math=math, random=random, Counter=Counter, datetime=datetime, bisect_left=bisect_left)
    for path, allowed in [(DATASET, {'get_preexisting_publications_for_author'}),
                          (SOURCE, {'_get_cited_references_for_author', 'predict_coauthors', '_build_author_first_pub_date'})]:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in allowed: nodes.append(node)
            elif path == SOURCE and isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
                ast.literal_eval(node.value); nodes.append(node)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), env)
    return env

class AuditedPaper(dict):
    def __init__(self, value, view, key, unavailable):
        super().__init__(value)
        self.view, self.key, self.unavailable = view, key, unavailable
    def get(self, key, default=None):
        if key == 'citation_trajectory' and self.unavailable:
            self.view.unavailable_bins[self.key] = self.unavailable
        return super().get(key, default)

class PaperView(Mapping):
    """Per-query input adapter. The upstream scorer's AST is not modified."""
    def __init__(self, papers, cutoff, historical=False, dedup=False):
        self.papers, self.cutoff = papers, cutoff
        self.cutday = date.fromisoformat(cutoff).toordinal()
        self.historical, self.dedup = historical, dedup
        self.cache = {}
        self.future_papers, self.unavailable_bins, self.changed_authors = set(), {}, set()

    def __iter__(self): return (p for p in self.papers if p in self)
    def __len__(self): return len(self.papers)
    def __contains__(self, key):
        p = self.papers.get(key)
        return p is not None and (not self.historical or p['date'] < self.cutoff)
    def __getitem__(self, key):
        p = self._lookup(key, direct=False)
        if p is None: raise KeyError(key)
        return p
    def get(self, key, default=None):
        p = self._lookup(key, direct=True)
        return default if p is None else p
    def _lookup(self, key, direct):
        p = self.papers.get(key)
        if p is None: return None
        if p['date'] >= self.cutoff:
            if direct:self.future_papers.add(key)
            if self.historical: return None
        if key in self.cache: return self.cache[key]
        # Upstream construction uses cumulative counts at publication+30*m days.
        # A bin ending on cutoff contains citations strictly before cutoff.
        bins = max(0, (self.cutday-p['_ordinal'])//30)
        trajectory = p['citation_trajectory']
        unavailable = len(trajectory)-bins if len(trajectory)>bins and any(trajectory[bins:]) else 0
        result = p
        if self.historical or self.dedup:
            result = dict(p)
            if self.historical: result['citation_trajectory'] = trajectory[:bins]
            if self.dedup:
                authors = list({a['author_id']: a for a in p['authors'] if a['author_id']}.values())
                if len(authors) != len(p['authors']): self.changed_authors.add(key)
                result['authors'] = authors
        result = AuditedPaper(result,self,key,unavailable)
        self.cache[key] = result
        return result

def metrics(pred, truth, new):
    truth, new = set(truth), set(new)
    if len(set(pred)) != len(pred): raise ValueError('Duplicate predictions')
    rel = np.array([a in truth for a in pred], dtype=float)
    def ndcg(k):
        size = min(k, len(rel)); discount = 1/np.log2(np.arange(size)+2)
        ideal = (1/np.log2(np.arange(min(k, len(truth)))+2)).sum()
        return float((rel[:size]*discount).sum()/ideal) if ideal else 0.
    repeat = truth-new
    return dict(ndcg1000=ndcg(1000), ndcg10=ndcg(10),
        rprecision=float(rel[:len(truth)].sum()/len(truth)),
        recall10=float(rel[:10].sum()/len(truth)), recall50=float(rel[:50].sum()/len(truth)),
        new_recall10=len(set(pred[:10])&new)/len(new) if new else None,
        repeat_recall10=len(set(pred[:10])&repeat)/len(repeat) if repeat else None,
        candidate_recall=len(set(pred)&truth)/len(truth))

def load_inputs():
    audit = read(STUDY/'results/final/prescience/data_audit.json')
    papers, author_objects = {}, {}
    intern = sys.intern
    pf = pq.ParquetFile(audit['source']['corpus'])
    for batch in pf.iter_batches(batch_size=512, columns=['corpus_id','date','authors','key_references','citation_trajectory']):
        for p in batch.to_pylist():
            cid = intern(str(p['corpus_id'])); authors = []
            for a in p['authors'] or []:
                aid = intern(str(a['author_id'])) if a.get('author_id') is not None else None
                if aid not in author_objects: author_objects[aid] = {'author_id':aid}
                authors.append(author_objects[aid])
            papers[cid] = dict(date=p['date'], _ordinal=date.fromisoformat(p['date']).toordinal(),
                authors=authors, citation_trajectory=p['citation_trajectory'] or [],
                key_references=[{'corpus_id':intern(str(r['corpus_id'])) if r.get('corpus_id') is not None else None}
                                for r in p['key_references'] or []])
    histories, clean_histories, first, unsorted = {}, {}, {}, 0
    with Path(audit['source']['author_history']).open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line); aid = intern(str(r['key']))
            pubs = [intern(str(x)) for x in r['value'] or [] if str(x) in papers]
            histories[aid] = pubs
            if not pubs: continue
            ds = [papers[p]['date'] for p in pubs]
            if ds != sorted(ds): unsorted += 1
            clean = sorted(set(pubs), key=lambda p:(papers[p]['date'],p))
            clean_histories[aid] = clean if clean != pubs else pubs
            first[aid] = min(ds)
    if unsorted: raise ValueError(f'Upstream binary-search assumption fails for {unsorted} histories')
    return papers, histories, clean_histories, first, audit

def baseline_rows():
    result = {}
    for seed in [42,43,44]:
        path = STUDY/f'results/final/prescience/full_lambdamart.s{seed}.predictions.json'
        rows = read(path)
        for r in rows:
            r['repeat_recall10'] = metrics(r['predicted'],r['truth'],r['new_truth'])['repeat_recall10']
            r.pop('predicted')
        result[seed] = {r['qid']:r for r in rows}
        del rows; gc.collect()
    ids = set(result[42])
    if len(ids) != 1000 or any(set(r)!=ids for r in result.values()): raise ValueError('Frozen query mismatch')
    return result

def freeze(baseline):
    evidence = [HERE/'REVIEW_PROTOCOL.md', Path(__file__), SOURCE, DATASET,
                HERE/'upstream/dataset/corpus/add_citation_metadata.py',
                STUDY/'final/global-lock.json', STUDY/'results/final/prescience/report.json']
    evidence += [STUDY/f'results/final/prescience/full_lambdamart.s{s}.predictions.json' for s in [42,43,44]]
    audit = read(STUDY/'results/final/prescience/data_audit.json')
    evidence += [Path(p) for p in audit['source'].values()]
    config = dict(design='Post-result fixed supplementary comparison; no test selection',
        tracks=TRACKS, k=1000, query_ids=sorted(baseline[42]), python=sys.version,
        hash_seed=os.environ.get('PYTHONHASHSEED'), upstream_commit='e54f6a5441aa389c18acfc11080facea6a95c1d5',
        sha256={str(p.relative_to(ROOT)):sha(p) for p in evidence},
        iterations=10000, inference_seed=314159)
    p = HERE/'supplementary-lock.json'
    if p.exists() and read(p)!=config: raise ValueError('Supplementary inputs changed; do not silently resume')
    if not p.exists(): write(p,config)
    return config

def summarize(baseline, config):
    import pytrec_eval
    results, contrasts, maximum_error = {}, {}, 0.
    for track in TRACKS:
        rows = [read(HERE/'checkpoints'/track/(q+'.json')) for q in sorted(baseline[42])]
        if len(rows)!=1000: raise ValueError('Incomplete track')
        qrels = {r['qid']:{a:1 for a in r['truth']} for r in rows}
        runs = {r['qid']:{a:float(len(r['predicted'])-i) for i,a in enumerate(r['predicted'])} for r in rows}
        independent = pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut.10,1000','recall.10,50','Rprec'}).evaluate(runs)
        for r in rows:
            for mine,other in [('ndcg1000','ndcg_cut_1000'),('ndcg10','ndcg_cut_10'),('rprecision','Rprec'),('recall10','recall_10'),('recall50','recall_50')]:
                maximum_error=max(maximum_error,abs(r[mine]-independent[r['qid']][other]))
        if maximum_error>1e-12: raise ValueError('Independent metric mismatch')
        aggregate = {k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in
            ['ndcg1000','ndcg10','rprecision','recall10','recall50','new_recall10','repeat_recall10','candidate_recall']}
        del runs,qrels,independent;gc.collect()
        paired = {}
        for metric in aggregate:
            eligible = [r for r in rows if r[metric] is not None]
            diffs = [np.mean([baseline[s][r['qid']][metric] for s in [42,43,44]])-r[metric] for r in eligible]
            paired[metric] = cluster_inference(diffs,[r['seed'] for r in eligible],repetitions=10000,seed=314159)
        contrasts[track] = paired['ndcg1000']
        results[track] = dict(metrics=aggregate,ranker_minus_comparator=paired,
            wall_seconds=sum(r['seconds'] for r in rows),
            queries_accessing_unavailable_citation_bins=sum(bool(r['audit']['unavailable_bin_papers']) for r in rows),
            queries_attempting_same_or_future_papers=sum(bool(r['audit']['future_papers']) for r in rows),
            queries_changed_author_lists=sum(bool(r['audit']['changed_authors']) for r in rows))
        write(HERE/(track+'.predictions.json'),rows)
        del rows;gc.collect()
    adjusted=holm({t:r['p_two_sided_cluster_signflip'] for t,r in contrasts.items()})
    for t in TRACKS:contrasts[t]['holm_p_three_supplementary_contrasts']=adjusted[t]
    report=dict(status='Complete supplementary post-result evaluation, not untouched confirmation',
        config=config,results=results,contrasts=contrasts,independent_metric_max_error=maximum_error,
        frozen_ranker_metrics={k:float(np.mean([r[k] for s in baseline.values() for r in s.values() if r[k] is not None])) for k in results[TRACKS[0]]['metrics']},
        limitations=['Published-input execution includes unavailable citation information; provenance sensitivity only.',
          'Historical trajectory mask relies on pinned release builder 30-day cumulative-bin convention; no independent event timestamps.',
          'Harmonized inputs change author deduplication and date-tie ordering.',
          'Fixed upstream padding uses Python hash/set behavior, pinned here to PYTHONHASHSEED=0.',
          'Earlier final results were known before these comparisons; all additional inference is supplementary.',
          'Cluster intervals condition on the selected cohort/models and do not model crossed network dependence.'])
    write(HERE/'official-comparator-results.json',report)
    print(json.dumps({'complete':True,'results':{k:v['metrics'] for k,v in results.items()},'contrasts':contrasts}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);ap.add_argument('--summarize-only',action='store_true');args=ap.parse_args()
    if os.environ.get('PYTHONHASHSEED')!='0':raise RuntimeError('Launch with PYTHONHASHSEED=0')
    baseline=baseline_rows();config=freeze(baseline)
    if args.summarize_only:return summarize(baseline,config)
    papers,histories,clean_histories,first,audit=load_inputs();env=official_functions()
    print(json.dumps({'loaded_papers':len(papers),'histories':len(histories),'rss_GB':psutil.Process().memory_info().rss/1e9}),flush=True)
    query_rows=sorted(baseline[42].values(),key=lambda q:(q['date'],q['qid']))
    if args.limit:query_rows=query_rows[:args.limit]
    # Reproduce upstream candidate insertion order: stable date order of histories.
    first_order=sorted(first.items(),key=lambda kv:kv[1]);eligible=set();cursor=0
    for qi,q in enumerate(query_rows):
        cutoff=(EPOCH+timedelta(days=q['date'])).isoformat()
        while cursor<len(first_order) and first_order[cursor][1]<cutoff:
            eligible.add(first_order[cursor][0]);cursor+=1
        for track in TRACKS:
            output=HERE/'checkpoints'/track/(q['qid']+'.json')
            if output.exists():
                saved=read(output)
                if saved['config_sha256']!=sha(HERE/'supplementary-lock.json'):raise ValueError('Checkpoint config mismatch')
                continue
            if shutil.disk_usage(HERE).free<1024**3 or psutil.virtual_memory().available<450*1024**2:
                raise RuntimeError('Resource reserve reached; safely resumable')
            view=PaperView(papers,cutoff,historical=track!='published_input',dedup=track=='harmonized_historical_input')
            history=clean_histories if track=='harmonized_historical_input' else histories
            start=time.perf_counter()
            predicted,scores=env['predict_coauthors'](q['seed'],cutoff,1000,eligible,history,view)
            seconds=time.perf_counter()-start
            if q['seed'] in predicted or any(a not in eligible for a in predicted):raise ValueError('Invalid candidate')
            if len(predicted)!=1000 or not np.isfinite(scores).all():raise ValueError('Invalid scorer output')
            row={k:q[k] for k in ['qid','seed','date','truth','new_truth','seed_history_count']}
            row.update(predicted=predicted,scores=scores,**metrics(predicted,q['truth'],q['new_truth']),seconds=seconds,
                config_sha256=sha(HERE/'supplementary-lock.json'),audit=dict(
                unavailable_bin_papers=len(view.unavailable_bins),future_papers=len(view.future_papers),changed_authors=len(view.changed_authors)))
            write(output,row);view.cache.clear();del view
        if (qi+1)%20==0: print(json.dumps({'queries':qi+1,'total':len(query_rows),'available_GB':psutil.virtual_memory().available/1e9}),flush=True)
    del papers,histories,clean_histories,first;gc.collect()
    if not args.limit:summarize(baseline,config)

if __name__=='__main__':main()
