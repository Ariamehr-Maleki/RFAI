"""Ensure deadline removal preserves NCN math and rejects changed resume settings."""
import ast
import copy
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
OLD=ast.parse((ROOT/'source-snapshots/ncn_temporal_time_limited.py').read_text())
NEW=ast.parse((ROOT/'scripts/ncn_temporal.py').read_text())

def function(tree,name):
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)

class ScheduleTests(unittest.TestCase):
    def test_training_loop_only_loses_date_guard(self):
        def loop(tree):
            return next(n for n in function(tree,'main').body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
        before=copy.deepcopy(loop(OLD));after=loop(NEW)
        removed=before.body.pop(0)
        self.assertIsInstance(removed,ast.If)
        self.assertIn('Authorized work window ended',ast.unparse(removed))
        self.assertEqual(ast.dump(before),ast.dump(after))
        self.assertEqual(ast.dump(function(OLD,'make_models')),ast.dump(function(NEW,'make_models')))

    def test_resume_migration_rejects_scientific_changes(self):
        nodes=[n for n in NEW.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LEGACY_TIME_LIMITED_SHA' for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='scheduling_only_migration')]
        namespace={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'migration','exec'),namespace)
        migrate=namespace['scheduling_only_migration']
        previous=dict(script_sha256=namespace['LEGACY_TIME_LIMITED_SHA'],seed=42,epochs=100,graph_cutoffs=[2016,2017])
        current=dict(previous,script_sha256='new-audited-code')
        self.assertTrue(migrate(previous,current))
        self.assertFalse(migrate(previous,dict(current,seed=43)))
        self.assertFalse(migrate(previous,dict(current,epochs=101)))
        self.assertFalse(migrate(dict(previous,script_sha256='unrecognized'),current))

if __name__=='__main__':unittest.main()
