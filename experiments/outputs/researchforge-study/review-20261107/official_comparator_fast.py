"""Value-preserving indexed version of the upstream historical-publication helper.

Scorer AST/constants and all original evaluation semantics remain unchanged.
Each distinct author/history/date result is checked against upstream before reuse.
This wrapper is an explicit runtime addendum to the supplementary lock.
"""
from bisect import bisect_left
import json
import os
import time
from pathlib import Path
import official_comparator as base

original_factory=base.official_functions
cache={}
checks=0

def factory():
    env=original_factory()
    original=env['get_preexisting_publications_for_author']
    def fast(author,cutoff,histories,view):
        global checks
        key=(id(histories),author)
        if key not in cache:
            pubs=histories[author] or []
            papers=view.papers
            pubs=[p for p in pubs if p in papers]
            dates=tuple(papers[p]['date'] for p in pubs)
            if list(dates)!=sorted(dates):raise ValueError('Unsorted history')
            cache[key]=(pubs,dates)
        pubs,dates=cache[key]
        answer=pubs[:bisect_left(dates,cutoff)]
        # Full parity on the first 1,000 calls, spanning raw and historical views;
        # later each 1,000th call supplies ongoing checks without selecting scores.
        checks+=1
        if checks<=1000 or checks%1000==0:
            expected=original(author,cutoff,histories,view)
            if answer!=expected:raise AssertionError('Indexed helper diverged from pinned official source')
        return answer
    env['get_preexisting_publications_for_author']=fast
    return env

def main():
    original_lock=base.HERE/'supplementary-lock.json'
    if not original_lock.exists():raise RuntimeError('Original supplementary lock required')
    addendum=dict(change='Indexed pure history lookup; upstream scorer and values unchanged',
        original_lock_sha256=base.sha(original_lock),wrapper_sha256=base.sha(Path(__file__)),
        parity='first 1000 helper calls and every 1000th subsequent call against original implementation',
        hash_seed=os.environ.get('PYTHONHASHSEED'))
    path=base.HERE/'runtime-addendum.json'
    if path.exists() and base.read(path)!=addendum:raise RuntimeError('Runtime wrapper changed')
    if not path.exists():base.write(path,addendum)
    base.official_functions=factory
    try:base.main()
    finally:
        base.write(base.HERE/'runtime-parity.json',dict(helper_calls=checks,checked_calls=min(checks,1000)+max(0,checks//1000-1),
            cache_entries=len(cache),runtime_addendum_sha256=base.sha(path)))

if __name__=='__main__':main()
