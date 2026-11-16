import unittest
from account_train_encoding_cost import account

class AccountingTests(unittest.TestCase):
    def test_contiguous_sessions_and_double_count_guard(self):
        def row(pid,n,seconds,rate):return dict(worker_pid=pid,completed_rows=n,session_seconds=seconds,session_docs_per_second=rate,config={},total_rows=30)
        sessions=[('first',[row(1,10,2,5)]),('second',[row(2,20,4,2.5),row(2,30,8,2.5)])]
        report=account(sessions,30,{})
        self.assertEqual(report['recorded_elapsed_seconds'],10);self.assertEqual(report['rows_per_recorded_second'],3)
        with self.assertRaisesRegex(ValueError,'Repeated'):account(sessions+[sessions[0]],30,{})
        with self.assertRaisesRegex(ValueError,'Overlapping or missing'):account(sessions[1:],30,{})

if __name__=='__main__':unittest.main()
