"""Development-only temporal graph LTR pilot on PreScience TRAIN parquet.

Does not read the official test parquet. This is not a final benchmark claim.
Content embeddings and the full official baseline suite are subsequent stages.
"""
import argparse
import bisect
import collections
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, LGBMRanker

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'work/data/prescience'
FEATURES = ['frequency_recent10', 'frequency_all', 'recency_collab',
            'common_neighbors_recent10', 'jaccard_recent10', 'degree_product_recent10',
            'seed_history_count', 'candidate_history_count', 'candidate_recent_degree',
            'topic_cosine_recent10', 'candidate_recency', 'history_balance',
            'frequency_reverse_recent10', 'seed_recent_degree']

def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)

def day(s):
    return int(np.datetime64(str(s)[:10], 'D').astype(int))

def unique_author_ids(authors):
    """One person contributes at most once per paper; preserve first-author order."""
    return tuple(dict.fromkeys(sys.intern(str(a['author_id'])) for a in (authors or []) if a.get('author_id')))

def load_data():
    papers = {}
    path = DATA / 'train.parquet'
    pf = pq.ParquetFile(path)
    cols = ['corpus_id', 'date', 'roles', 'authors', 'categories']
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        for p in batch.to_pylist():
            if not p.get('date'):
                continue
            aid = unique_author_ids(p.get('authors'))
            # Share equal identifier/category strings across the historical corpus.
            # Values and insertion order are unchanged; avoids many duplicate objects.
            papers[sys.intern(str(p['corpus_id']))] = dict(date=day(p['date']), authors=aid,
                target='target' in (p.get('roles') or []), categories=tuple(sys.intern(c) for c in (p.get('categories') or [])))
    histories, dates = {}, {}
    with (DATA / 'author_publications.jsonl').open(encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            ids = sorted({sys.intern(str(i)) for i in (x['value'] or []) if str(i) in papers},
                         key=lambda i: (papers[i]['date'], i))
            if ids:
                key=sys.intern(str(x['key']))
                histories[key] = ids
                dates[key] = [papers[i]['date'] for i in ids]
    log(f'Loaded {len(papers)} training-corpus papers, {len(histories)} author histories')
    return papers, histories, dates

def make_queries(papers, histories, dates):
    queries = []
    for cid, p in papers.items():
        if not p['target']:
            continue
        authors = list(dict.fromkeys(a for a in p['authors']
            if a in dates and dates[a][0] < p['date']))
        if len(authors) >= 2:
            queries.append(dict(qid=cid, date=p['date'], seed=authors[0], truth=authors[1:]))
    return sorted(queries, key=lambda q: (q['date'], q['qid']))

def sample_queries(queries, n, seed):
    if not n or n >= len(queries):
        return queries
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(queries), n, replace=False))
    return [queries[i] for i in idx]

def build_features(queries, papers, histories, dates, n_candidates, seed, training):
    author_ids = sorted(histories)
    earliest = np.array([dates[a][0] for a in author_ids])
    X, y, groups, metadata = [], [], [], []
    injected = 0
    for qi, q in enumerate(queries):
        cutoff, source = q['date'], q['seed']
        cache = {}

        def profile(a):
            if a in cache:
                return cache[a]
            end = bisect.bisect_left(dates.get(a, []), cutoff)
            ids = histories.get(a, [])[:end]
            recent, allfreq, last, topics = collections.Counter(), collections.Counter(), {}, collections.Counter()
            # Only the seed needs its full coauthor history. Candidate all-history
            # frequencies are not features; avoid repeatedly traversing them.
            for pid in (ids if a == source else ids[-10:]):
                p = papers[pid]
                assert p['date'] < cutoff
                for other in p['authors']:
                    if other != a:
                        allfreq[other] += 1
                        last[other] = max(last.get(other, -1000000), p['date'])
            for pid in ids[-10:]:
                p = papers[pid]
                recent.update(other for other in p['authors'] if other != a)
                topics.update(p['categories'])
            result = (recent, allfreq, last, topics, end, papers[ids[-1]]['date'] if ids else -1000000)
            cache[a] = result
            return result

        sf, sa, sl, st, sn, _ = profile(source)
        # Retrieval sees only pre-query histories, never the relevance labels.
        retrieval = collections.Counter()
        direct = sorted(sa, key=lambda a: (-sa[a], a))
        for a in direct[:30]:
            retrieval.update(profile(a)[0])
        ordered = direct + sorted(retrieval, key=lambda a: (-retrieval[a], a))
        candidates, used = [], {source}
        for a in ordered:
            if a not in used and a in dates and dates[a][0] < cutoff:
                candidates.append(a)
                used.add(a)
                if len(candidates) == n_candidates:
                    break
        qseed = int(hashlib.sha256(f'{seed}:{q["qid"]}'.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(qseed)
        eligible = np.flatnonzero(earliest < cutoff)
        if len(candidates) < n_candidates:
            for j in rng.permutation(eligible):
                a = author_ids[j]
                if a not in used:
                    candidates.append(a)
                    used.add(a)
                    if len(candidates) == n_candidates:
                        break
        recall = len(set(candidates) & set(q['truth'])) / len(q['truth'])
        if training:
            for a in q['truth']:
                if a not in used:
                    candidates.append(a)
                    used.add(a)
                    injected += 1
        # Sort identities before scoring: deterministic, feature-independent ties.
        candidates.sort()
        sx = set(sf)
        rows = []
        for a in candidates:
            cf, ca, cl, ct, cn, last_pub = profile(a)
            cx = set(cf)
            inter = len(sx & cx)
            dot = sum(v * ct[k] for k, v in st.items())
            denom = math.sqrt(sum(v*v for v in st.values()) * sum(v*v for v in ct.values()))
            rows.append([sf[a], sa[a], 1/(1+cutoff-sl[a]) if a in sl else 0,
                         inter, inter/max(1,len(sx|cx)), math.log1p(len(sx)*len(cx)),
                         math.log1p(sn), math.log1p(cn), math.log1p(len(cx)),
                         dot/denom if denom else 0, 1/(1+cutoff-last_pub),
                         min(sn,cn)/max(1,sn,cn), cf[source], math.log1p(len(sx))])
        X.extend(rows)
        y.extend(int(a in q['truth']) for a in candidates)
        groups.append(len(candidates))
        metadata.append(dict(**q, candidates=candidates, candidate_recall=recall,
                             new_truth=[a for a in q['truth'] if sa[a] == 0]))
        if qi % 10 == 0:
            log(f'{"train" if training else "validation"} queries {qi+1}/{len(queries)}')
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8), np.asarray(groups), metadata, injected

def rank_metrics(pred, truth, new_truth=()):
    truth = set(truth)
    rel = np.array([a in truth for a in pred], dtype=float)
    def ndcg(k):
        top = rel[:k]
        dcg = np.sum(top / np.log2(np.arange(len(top))+2))
        ideal = np.sum(1/np.log2(np.arange(min(k,len(truth)))+2))
        return float(dcg/ideal) if ideal else 0.
    r = len(truth)
    new = set(new_truth)
    return dict(ndcg10=ndcg(10), ndcg1000=ndcg(1000),
                rprecision=float(rel[:r].sum()/r) if r else 0.,
                recall10=float(rel[:10].sum()/r) if r else 0.,
                recall50=float(rel[:50].sum()/r) if r else 0.,
                new_recall10=len(set(pred[:10]) & new)/len(new) if new else None)

def evaluate(scores, groups, meta, save_predictions=False):
    rows, start = [], 0
    for n, q in zip(groups, meta):
        order = np.argsort(-scores[start:start+n], kind='stable')
        pred = [q['candidates'][i] for i in order]
        row = dict(qid=q['qid'], seed=q['seed'], candidate_recall=q['candidate_recall'],
                   **rank_metrics(pred,q['truth'],q['new_truth']))
        if save_predictions:
            row.update(predicted=pred, truth=q['truth'], new_truth=q['new_truth'])
        rows.append(row)
        start += n
    keys = ['ndcg10','ndcg1000','rprecision','recall10','recall50','new_recall10','candidate_recall']
    agg = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
           if any(r[k] is not None for r in rows) else None for k in keys}
    return agg, rows

def cluster_difference(best_rows, base_rows, seed=42, repeats=2000):
    assert [r['qid'] for r in best_rows] == [r['qid'] for r in base_rows]
    clusters = collections.defaultdict(list)
    for a,b in zip(best_rows,base_rows):
        clusters[a['seed']].append(a['ndcg10']-b['ndcg10'])
    vals = list(clusters.values())
    sums, counts = np.array([sum(x) for x in vals]), np.array([len(x) for x in vals])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0,len(vals),size=(repeats,len(vals)))
    delta = sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
    return dict(delta=float(sums.sum()/counts.sum()), ci95=np.quantile(delta,[.025,.975]).tolist(),
                clusters=len(vals), note='Development validation CI; configuration was selected here, not confirmatory inference.')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-queries',type=int,default=300)
    ap.add_argument('--valid-queries',type=int,default=100)
    ap.add_argument('--candidates',type=int,default=200)
    ap.add_argument('--seed',type=int,default=42)
    args = ap.parse_args()
    began = time.perf_counter()
    run = OUT / 'results' / f'pilot-dedup-{args.train_queries}-{args.valid_queries}-{args.candidates}-s{args.seed}'
    run.mkdir(parents=True,exist_ok=True)
    papers, histories, dates = load_data()
    queries = make_queries(papers,histories,dates)
    split = day('2024-07-01')
    train_all = [q for q in queries if q['date'] < split]
    valid_all = [q for q in queries if q['date'] >= split]
    assert train_all and valid_all
    train = sample_queries(train_all,args.train_queries,args.seed)
    valid = sample_queries(valid_all,args.valid_queries,args.seed)
    log(f'Eligible train={len(train_all)}, validation={len(valid_all)}; pilot={len(train)}/{len(valid)}')
    tx,ty,tg,tm,injected=build_features(train,papers,histories,dates,args.candidates,args.seed,True)
    vx,vy,vg,vm,_=build_features(valid,papers,histories,dates,args.candidates,args.seed,False)
    np.savez_compressed(run/'features.npz',train_X=tx,train_y=ty,train_group=tg,valid_X=vx,valid_y=vy,valid_group=vg)
    (run/'validation_queries.json').write_text(json.dumps(vm),encoding='utf-8')
    (run/'train_query_ids.json').write_text(json.dumps([q['qid'] for q in train]),encoding='utf-8')
    predictions = {'frequency_recent10':vx[:,0], 'frequency_all':vx[:,1],
                   'common_neighbors_recent10':vx[:,3], 'topic_cosine_recent10':vx[:,9],
                   'random':np.random.default_rng(args.seed).random(len(vx))}
    tuning=[]
    for family in ['lambdamart','binary_lightgbm']:
        best=None
        for leaves,estimators in [(7,60),(15,100),(31,100)]:
            cls=LGBMRanker if family=='lambdamart' else LGBMClassifier
            model=cls(n_estimators=estimators,num_leaves=leaves,learning_rate=.05,
                      reg_lambda=1,random_state=args.seed,n_jobs=2,verbosity=-1,deterministic=True,
                      force_col_wise=True,objective='lambdarank' if family=='lambdamart' else 'binary')
            before=time.perf_counter()
            model.fit(tx,ty,**({'group':tg} if family=='lambdamart' else {}))
            scores=model.predict(vx) if family=='lambdamart' else model.predict_proba(vx)[:,1]
            agg,_=evaluate(scores,vg,vm)
            tuning.append(dict(family=family,num_leaves=leaves,n_estimators=estimators,
                               fit_predict_seconds=time.perf_counter()-before,**agg))
            if best is None or agg['ndcg10']>best[0]:
                best=(agg['ndcg10'],scores,model)
        predictions[family]=best[1]
        best[2].booster_.save_model(str(run/f'{family}.txt'))
    metrics, per_method={},{}
    for name,scores in predictions.items():
        agg,rows=evaluate(scores,vg,vm,True)
        metrics[name]=agg
        per_method[name]=rows
        (run/f'{name}.predictions.json').write_text(json.dumps(rows),encoding='utf-8')
    report=dict(status='development pilot; NOT official test results', config=vars(args),
        features=FEATURES, full_eligible_train=len(train_all),full_eligible_validation=len(valid_all),
        training_positive_injections=injected, metrics=metrics, tuning=tuning,
        paired_validation_delta=cluster_difference(per_method['lambdamart'],per_method['frequency_all']),
        wall_seconds=time.perf_counter()-began,
        limitations=['Graph/topic pilot; no publication text embeddings yet.',
                     'Restricted retrieved candidates; not comparable to full-pool published results.',
                     'Candidate histories restricted to TRAIN corpus; official TEST parquet never read.',
                     'Validation used for model selection; uncertainty is exploratory.',
                     'Observed absent collaborations are not verified unsuitable matches.'])
    (run/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    log(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
