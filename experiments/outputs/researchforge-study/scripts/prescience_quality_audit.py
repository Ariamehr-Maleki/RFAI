"""Streamed TRAIN-only data integrity audit; no official test access."""
import collections
import hashlib
import json
import re
from datetime import date
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/researchforge-study/data-audit'
SOURCE=ROOT/'work/data/prescience/train.parquet'

def main():
    count=collections.Counter();ids=collections.Counter();titles=collections.Counter()
    dates={};roles=collections.Counter();minimum=None;maximum=None
    columns=['corpus_id','date','roles','title','abstract','authors']
    for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=512,columns=columns):
        for p in batch.to_pylist():
            count['rows']+=1;cid=p['corpus_id'];ids[cid]+=1
            roles.update(p['roles'] or [])
            count['missing_corpus_id']+=not bool(cid)
            text=re.sub(r'\s+',' ',(p['title'] or '').strip().casefold())
            if text:titles[hashlib.sha256(text.encode()).hexdigest()]+=1
            count['empty_title']+=not bool(text)
            count['empty_abstract']+=not bool((p['abstract'] or '').strip())
            aa=[a['author_id'] for a in p['authors'] or [] if a.get('author_id')]
            count['papers_without_author_ids']+=not bool(aa)
            count['duplicate_author_id_papers']+=len(aa)!=len(set(aa))
            if not p['date']:
                count['missing_date']+=1;continue
            try:d=date.fromisoformat(str(p['date'])[:10])
            except ValueError:count['invalid_date']+=1;continue
            dates[cid]=d
            minimum=d if minimum is None else min(minimum,d)
            maximum=d if maximum is None else max(maximum,d)
            if 'target' in (p['roles'] or []):
                count['targets_outside_declared_train_window']+=not (date(2023,10,1)<=d<date(2024,10,1))
    refstats=collections.Counter()
    for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=1024,columns=['corpus_id','key_references']):
        for p in batch.to_pylist():
            for ref in p['key_references'] or []:
                rid=ref['corpus_id'];refstats['reference_edges']+=1
                if rid not in dates:refstats['reference_missing_from_train']+=1
                elif p['corpus_id'] in dates:
                    if dates[rid]>dates[p['corpus_id']]:refstats['reference_date_after_citing_date']+=1
                    elif dates[rid]==dates[p['corpus_id']]:refstats['reference_same_day']+=1
    result=dict(source='PreScience TRAIN only',counts=dict(count),roles=dict(roles),
        unique_corpus_ids=len(ids),duplicate_id_extra_rows=sum(v-1 for v in ids.values()),
        repeated_normalized_title_groups=sum(v>1 for v in titles.values()),
        repeated_normalized_title_extra_rows=sum(v-1 for v in titles.values()),
        min_date=str(minimum),max_date=str(maximum),references=dict(refstats),
        v2_paper_reported_train_rows=373707,
        difference_to_v2_reported_train_rows=count['rows']-373707,
        caveats=['Repeated normalized titles alone do not prove duplicate papers.',
                 'Reference date anomalies can reflect preprint/version metadata, not necessarily invalid citations.',
                 'The ranker already excludes same-day and future referenced papers at each query cutoff.',
                 'No test-set integrity or outcomes are inferred from this TRAIN-only audit.'])
    OUT.mkdir(exist_ok=True);(OUT/'prescience_train_integrity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
