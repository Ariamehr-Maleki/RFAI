"""BM25 on hashed terms, not an exact Lucene/Pyserini reproduction.

Positive IDF log(1+(N-df+.5)/(df+.5)), k1=1.2,b=.75. Query terms binary.
No quantized length norms. Fixed candidate-corpus statistics, no qrels.
"""
import numpy as np

class HashedBM25:
    def __init__(self,normalized_csr,norms,lengths,k1=1.2,b=.75):
        self.postings=normalized_csr.tocsc()
        self.norms=np.asarray(norms);self.lengths=np.asarray(lengths)
        self.k1=k1;self.b=b
        self.n=normalized_csr.shape[0]
        df=np.diff(self.postings.indptr)
        self.idf=np.log1p((self.n-df+.5)/(df+.5))
        self.denominator=k1*(1-b+b*self.lengths/max(float(self.lengths.mean()),1e-12))

    def score(self,query_terms):
        scores=np.zeros(self.n,dtype=np.float32)
        for term in np.unique(query_terms):
            start,end=self.postings.indptr[term:term+2]
            rows=self.postings.indices[start:end]
            tf=np.rint(self.postings.data[start:end]*self.norms[rows])
            scores[rows]+=self.idf[term]*tf*(self.k1+1)/(tf+self.denominator[rows])
        return scores
