import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('review_cli',ROOT/'src/rfai_review/cli.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
class ReviewerTests(unittest.TestCase):
    def test_released_numbers_preserve_negative_result(self):
        s=cli.summary();p=s['PreScience nDCG@1000']
        self.assertGreater(p['official_harmonized'],p['ranker'])
        self.assertGreater(p['ranker'],p['matched_classifier'])
        self.assertLess(s['ogbl-collab Hits@50']['graph__lambdamart'],s['ogbl-collab Hits@50']['ncn'])
    def test_missing_and_modified_payloads_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'MANIFEST.json').write_text(json.dumps({'files':{'payload.txt':{'bytes':1,'sha256':'bad'}}}))
            with self.assertRaisesRegex(ValueError,'Missing payload'):cli.audit(p)
            (p/'payload.txt').write_text('x')
            with self.assertRaisesRegex(ValueError,'Changed payload'):cli.audit(p)
    def test_manifest_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'MANIFEST.json').write_text(json.dumps({'files':{'../escape.txt':{'bytes':0,'sha256':'bad'}}}))
            with self.assertRaisesRegex(ValueError,'escapes repository'):cli.audit(p)
if __name__=='__main__':unittest.main()
