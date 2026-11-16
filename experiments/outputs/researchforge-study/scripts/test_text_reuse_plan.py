"""Protect label-column exclusion and reject stale text cache reuse."""
import unittest
from unittest.mock import patch
from prescience_text_reuse_plan import text_hash,reuse_row,text_rows,COLUMNS


class ReuseTests(unittest.TestCase):
    def test_reuse_requires_id_and_exact_encoder_input(self):
        row=dict(corpus_id='a',title='Title',abstract='Text')
        train={'a':(7,text_hash(row))}
        self.assertEqual(reuse_row(row,train),7)
        self.assertEqual(reuse_row(dict(row,title='Changed'),train),-1)
        self.assertEqual(reuse_row(dict(row,corpus_id='b'),train),-1)
        self.assertEqual(reuse_row(dict(row,authors=['secret']),train),7)

    def test_reader_requests_only_nonlabel_columns(self):
        class FakeFile:
            def iter_batches(self,*,batch_size,columns):
                if columns!=['corpus_id','title','abstract']:raise AssertionError('Forbidden test column')
                return []
        with patch('prescience_text_reuse_plan.pq.ParquetFile',return_value=FakeFile()):
            self.assertEqual(list(text_rows('synthetic')),[])
        self.assertEqual(COLUMNS,['corpus_id','title','abstract'])


if __name__=='__main__':unittest.main()
