"""Explicit alternative interpretations of the PreScience v2 fixed scorer.

All variants are description-based sensitivity comparators, not official code.
Default reproduces the existing cleaned QueryEngine's approximate scorer.
"""
import bisect
import collections

VARIANTS={
    'default':dict(normalize=True,citation_weight=1.,third_damping=.3),
    'citation_off':dict(normalize=True,citation_weight=0.,third_damping=.3),
    'citation_quarter':dict(normalize=True,citation_weight=.25,third_damping=.3),
    'citation_four':dict(normalize=True,citation_weight=4.,third_damping=.3),
    'independent_hop3':dict(normalize=True,citation_weight=1.,third_damping=.5),
    'raw_transitions':dict(normalize=False,citation_weight=1.,third_damping=.3),
}


class FixedScorer:
    def __init__(self,papers,histories,dates,source,cutoff):
        self.papers,self.histories,self.dates=papers,histories,dates
        self.source,self.cutoff=source,cutoff
        self.cache={}

    def profile(self,author):
        if author not in self.cache:
            end=bisect.bisect_left(self.dates.get(author,[]),self.cutoff)
            ids=self.histories.get(author,[])[:end]
            counts=collections.Counter()
            for pid in ids[-45:]:
                counts.update(a for a in self.papers[pid]['authors'] if a!=author)
            self.cache[author]=(counts,ids)
        return self.cache[author]

    def score(self,normalize=True,citation_weight=1.,third_damping=.3):
        direct,history=self.profile(self.source)
        ordered=lambda x:sorted(x,key=lambda a:(-x[a],a))
        hop2,hop3=collections.Counter(),collections.Counter()
        for author in ordered(direct)[:150]:
            outgoing,_=self.profile(author)
            scale=.6*direct[author]/(max(1,sum(outgoing.values())) if normalize else 1)
            for a,count in outgoing.items():hop2[a]+=scale*count
        hop2.pop(self.source,None)
        for author in ordered(hop2)[:100]:
            outgoing,_=self.profile(author)
            scale=third_damping*hop2[author]/(max(1,sum(outgoing.values())) if normalize else 1)
            for a,count in outgoing.items():hop3[a]+=scale*count
        cited={}
        for pid in history:
            paper=self.papers[pid]
            for ref in paper.get('refs',()):
                if ref in self.papers and self.papers[ref]['date']<self.cutoff:
                    cited[ref]=max(cited.get(ref,-10**6),paper['date'])
        refs=sorted(cited,key=lambda p:(-cited[p],-self.papers[p]['date'],p))[:10]
        citation=collections.Counter()
        for pid in refs:
            for a in self.papers[pid]['authors']:citation[a]+=citation_weight
        scores=direct+hop2+hop3+citation
        scores.pop(self.source,None)
        for a in list(scores):
            end=bisect.bisect_left(self.dates.get(a,[]),self.cutoff)
            if not end:del scores[a]
            else:scores[a]*=2**(-(self.cutoff-self.dates[a][end-1])/(2.5*365.25))
        return scores
