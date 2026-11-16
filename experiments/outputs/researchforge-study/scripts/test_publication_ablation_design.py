import ast
from pathlib import Path
import unittest
from publication_ablation_design import factorial_columns,history_group,CONTROLS,COMPONENTS,COMPOSITES

def feature_names():
    tree=ast.parse(Path(__file__).with_name('prescience_hybrid_graph.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FEATURES' for t in n.targets))
    return ast.literal_eval(node.value)

class AblationTests(unittest.TestCase):
    def test_all_cells_exclude_mixed_evidence_and_exactly_partition_features(self):
        names=feature_names();cells=factorial_columns(names)
        self.assertEqual(len(cells),8)
        for label,cols in cells.items():
            selected={names[i] for i in cols}
            self.assertFalse(selected&COMPOSITES)
            self.assertTrue(CONTROLS<=selected)
            for component,features in COMPONENTS.items():
                self.assertEqual(selected&features,features if component in label.split('+') else set())
        self.assertNotIn(names.index('category_cosine'),cells['network+explicit_time'])
        self.assertEqual(len(cells['text+network+explicit_time']),21)
    def test_unclassified_feature_rejected(self):
        with self.assertRaises(ValueError):factorial_columns(feature_names()+['new_hidden_signal'])
    def test_group_boundaries(self):
        self.assertEqual([history_group(n) for n in [1,3,4,10,11]],['1-3','1-3','4-10','4-10','11+'])
        with self.assertRaises(ValueError):history_group(0)

if __name__=='__main__':unittest.main()
