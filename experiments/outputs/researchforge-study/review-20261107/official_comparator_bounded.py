"""Value-preserving bounded cache for the original upstream history helper.

An operational memory addendum; no scoring constants, candidates or metrics change.
"""
from bisect import bisect_left
from collections import OrderedDict
from pathlib import Path
import os
import official_comparator as base

original_factory=base.official_functions
MAX_CACHED_AUTHORS=10000
MAX_CACHED_PUBLICATIONS=1000000
cache=OrderedDict()
cached_publications=0
checks=0
evictions=0
peak_authors=0
peak_publications=0

def factory():
    env=original_factory()
    original=env['get_preexisting_publications_for_author']
    def bounded(author,cutoff,histories,view):
        global checks,cached_publications,evictions,peak_authors,peak_publications
        key=(id(histories),author)
        if key in cache:
            pubs,dates=cache[key]
            cache.move_to_end(key)
        else:
            pubs=[p for p in (histories[author] or []) if p in view.papers]
            dates=tuple(view.papers[p]['date'] for p in pubs)
            if list(dates)!=sorted(dates):raise ValueError('Unsorted history')
            if len(pubs)<=MAX_CACHED_PUBLICATIONS:
                while cache and (len(cache)>=MAX_CACHED_AUTHORS or cached_publications+len(pubs)>MAX_CACHED_PUBLICATIONS):
                    _,old=cache.popitem(last=False)
                    cached_publications-=len(old[0]);evictions+=1
                cache[key]=(pubs,dates);cached_publications+=len(pubs)
                peak_authors=max(peak_authors,len(cache));peak_publications=max(peak_publications,cached_publications)
        answer=pubs[:bisect_left(dates,cutoff)]
        checks+=1
        if checks<=1000 or checks%1000==0:
            if answer!=original(author,cutoff,histories,view):raise AssertionError('Bounded helper diverged from upstream')
        return answer
    env['get_preexisting_publications_for_author']=bounded
    return env

def main():
    path=base.HERE/'runtime-bounded-addendum.json'
    config=dict(change='Bounded LRU history cache; identical ordered helper values and unchanged scorer',
        original_lock_sha256=base.sha(base.HERE/'supplementary-lock.json'),
        previous_runtime_addendum_sha256=base.sha(base.HERE/'runtime-addendum.json'),
        wrapper_sha256=base.sha(Path(__file__)),hash_seed=os.environ.get('PYTHONHASHSEED'),
        max_cached_authors=MAX_CACHED_AUTHORS,max_cached_publications=MAX_CACHED_PUBLICATIONS,
        parity='first 1000 helper calls and every 1000th subsequent call against original implementation')
    if path.exists() and base.read(path)!=config:raise RuntimeError('Bounded wrapper changed')
    if not path.exists():base.write(path,config)
    base.official_functions=factory
    try:base.main()
    finally:
        base.write(base.HERE/'runtime-bounded-parity.json',dict(helper_calls=checks,
            checked_calls=min(checks,1000)+max(0,checks//1000-1),cache_entries=len(cache),
            cached_publications=cached_publications,evictions=evictions,peak_authors=peak_authors,
            peak_publications=peak_publications,runtime_addendum_sha256=base.sha(path)))

if __name__=='__main__':main()
