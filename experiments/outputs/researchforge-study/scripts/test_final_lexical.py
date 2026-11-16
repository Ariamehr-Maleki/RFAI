import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer
from prescience_final_lexical import build,COLUMNS
from text_index import TextIndex


class LexicalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]/'work');self.root=Path(self.tmp.name)
        self.source=self.root/'corpus.parquet';self.output=self.root/'index'
        self.rows=[dict(corpus_id=str(i),date=f'2024-01-0{i+1}',title='alpha science',abstract='beta' if i else 'gamma',
                        authors=['forbidden'],roles=['forbidden'],key_references=['forbidden']) for i in range(4)]
        pq.write_table(pa.Table.from_pylist(self.rows),self.source)
    def tearDown(self):self.tmp.cleanup()
    def test_exact_transform_masks_and_forbidden_columns(self):
        factory=pq.ParquetFile;calls=[]
        class Reader:
            def __init__(self,path):self.reader=factory(path)
            def iter_batches(self,**kwargs):calls.append(kwargs['columns']);return self.reader.iter_batches(**kwargs)
        with patch('prescience_final_lexical.pq.ParquetFile',Reader):build(self.source,self.output,batch_size=2)
        self.assertEqual(calls,[COLUMNS]);index=TextIndex(self.output)
        expected=HashingVectorizer(n_features=2**18,alternate_sign=False,norm='l2',stop_words='english',dtype=np.float32).transform(
            [r['title']+' '+r['abstract'] for r in self.rows])
        self.assertEqual((index.matrix-expected).nnz,0)
        cutoff=int(np.datetime64('2024-01-03','D').astype(int))
        scores=index.similarities(['0','3'],cutoff);self.assertTrue(np.all(scores[2:]==0));self.assertGreater(scores[0],0)
    def test_interrupted_chunks_resume_and_corruption_rejected(self):
        def interrupt(number):
            if number==0:raise RuntimeError('synthetic interruption')
        with self.assertRaises(RuntimeError):build(self.source,self.output,2,after_chunk=interrupt)
        report=build(self.source,self.output,2,expected_ids=['0','1','2','3']);self.assertEqual(report['rows'],4)
        self.assertEqual(report,build(self.source,self.output,2))
        with (self.output/'data.bin').open('ab') as stream:stream.write(b'bad')
        with self.assertRaisesRegex(ValueError,'file changed'):build(self.source,self.output,2)
    def test_committed_chunk_tampering_rejected(self):
        def interrupt(number):raise RuntimeError('synthetic interruption')
        with self.assertRaises(RuntimeError):build(self.source,self.output,2,after_chunk=interrupt)
        with (self.output/'chunks/000000.npz').open('ab') as stream:stream.write(b'bad')
        with self.assertRaisesRegex(ValueError,'chunk changed'):build(self.source,self.output,2)


if __name__=='__main__':unittest.main()
