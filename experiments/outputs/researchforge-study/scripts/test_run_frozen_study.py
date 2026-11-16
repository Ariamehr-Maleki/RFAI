"""Scheduling fixtures only; mock execution never invokes actual final runners."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_frozen_study as chain
from frozen_ranker import digest


class FinalChainTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='final-chain-fixture-',dir=chain.WORK)
        self.root=Path(self.tmp.name);self.study=self.root/'study';self.work=self.root/'work';self.work.mkdir()
        self.state=self.work/'state.json';self.writer=self.work/'writer.lock';self.path=self.root/'fixture-lock.json'
        self.patch=patch.multiple(chain,ROOT=self.root,STUDY=self.study,WORK=self.work,STATE=self.state,WRITER=self.writer)
        self.patch.start();self.calls=[]
        self.lock=dict(lock_sha256='fixture',payload={'plan':{'datasets':{name:{'output_directory':'result/'+name} for name,_ in chain.RUNNERS}}})
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def execute(self,label,command,pinned,**kw):
        self.calls.append(label)
        if label in ['final-tables','final-figures']:
            folder=Path(command[command.index('--output')+1]);folder.mkdir(parents=True)
            (folder/'payload.txt').write_text('fixture')
            status='Verified frozen-result table export' if label=='final-tables' else 'Verified frozen-result figures'
            (folder/'provenance.json').write_text(json.dumps(dict(status=status,lock_sha256='fixture',files={'payload.txt':digest(folder/'payload.txt')})))
        else:
            dataset=label.removeprefix('final-');folder=self.root/'result'/dataset;folder.mkdir(parents=True)
            (folder/'report.json').write_text(json.dumps(dict(run_id='fixture')))
    def checked(self,root,lock,dataset):
        path=root/'result'/dataset/'report.json';return path,json.loads(path.read_text())
    def run_mocked(self):
        with patch.object(chain,'verify_lock',return_value=self.lock),patch.object(chain,'checked_report',side_effect=self.checked),patch.object(chain,'execute',side_effect=self.execute):
            chain.run(self.path,'fixture')
    def test_missing_lock_never_launches_and_wait_has_no_test_access(self):
        with patch.object(chain,'execute') as execute,patch.object(chain,'verify_lock') as verify:
            with self.assertRaises(FileNotFoundError):chain.run(self.path,'fixture')
            execute.assert_not_called();verify.assert_not_called()
            with patch.object(chain.time,'sleep',side_effect=RuntimeError('fixture wait interruption')):
                with self.assertRaisesRegex(RuntimeError,'wait interruption'):chain.run(self.path,'fixture',True)
            execute.assert_not_called();verify.assert_not_called()
        self.assertFalse(self.writer.exists())
    def test_serial_order_resume_and_changed_export(self):
        self.path.write_text('fixture');self.run_mocked()
        self.assertEqual(self.calls,['final-ogbl-collab','final-exHarmony','final-prescience','final-tables','final-figures'])
        self.assertIn('manuscript',json.loads(self.state.read_text())['stage'])
        self.calls.clear();self.run_mocked();self.assertEqual(self.calls,[])
        (self.study/'results/final/tables/payload.txt').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'Changed export'):self.run_mocked()
    def test_different_existing_run_id_rejected(self):
        self.path.write_text('fixture');self.run_mocked();self.calls.clear()
        (self.root/'result/ogbl-collab/report.json').write_text(json.dumps(dict(run_id='different')))
        with self.assertRaisesRegex(ValueError,'another run ID'):self.run_mocked()
        self.assertEqual(self.calls,[])
    def test_partial_export_preserved_for_inspection(self):
        self.path.write_text('fixture');folder=self.study/'results/final/tables';folder.mkdir(parents=True)
        (folder/'partial.txt').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'Partial export'):self.run_mocked()
        self.assertEqual((folder/'partial.txt').read_text(),'preserve')


if __name__=='__main__':unittest.main()
