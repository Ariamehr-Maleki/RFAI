import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from supervisor_state import atomic_state


class StateTests(unittest.TestCase):
    def test_transient_conflict_retries_without_destroying_prior_state(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]/'work') as tmp:
            path=Path(tmp)/'state.json';path.write_text('{"stage":"old"}')
            replace=os.replace;calls=[]
            def flaky(source,destination):
                calls.append(1)
                if len(calls)<3:
                    self.assertEqual(json.loads(path.read_text())['stage'],'old')
                    raise PermissionError('Windows sharing conflict')
                return replace(source,destination)
            with patch('supervisor_state.os.replace',side_effect=flaky),patch('supervisor_state.time.sleep'):
                atomic_state(path,dict(stage='new'))
            self.assertEqual(len(calls),3);self.assertEqual(json.loads(path.read_text())['stage'],'new')
            self.assertEqual(list(Path(tmp).glob('*.tmp')),[])
    def test_permanent_failure_preserves_old_file_and_raises(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]/'work') as tmp:
            path=Path(tmp)/'state.json';path.write_text('{"stage":"old"}')
            with patch('supervisor_state.os.replace',side_effect=PermissionError('denied')),patch('supervisor_state.time.sleep'):
                with self.assertRaises(PermissionError):atomic_state(path,dict(stage='new'),attempts=3)
            self.assertEqual(json.loads(path.read_text())['stage'],'old')
            self.assertEqual(list(Path(tmp).glob('*.tmp')),[])


if __name__=='__main__':unittest.main()
