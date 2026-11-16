import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path
from build_reproducibility_package import ROOT,archive_files,verify_archive,build,digest,STATUSES

class PackageTests(unittest.TestCase):
    def test_streamed_archive_is_exact_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as directory:
            folder=Path(directory);source=folder/'negative-result.txt';source.write_text('delta = -0.02\n',encoding='utf-8');dest=folder/'fixture.zip'
            archive_files(ROOT,[source],dest,{'status':'synthetic software fixture'},reserve=0)
            record=verify_archive(dest);self.assertEqual(len(record['files']),1)
            self.assertEqual(next(iter(record['files'].values()))['bytes'],source.stat().st_size)
            with self.assertRaisesRegex(ValueError,'New workspace'):archive_files(ROOT,[source],dest,{},reserve=0)
    def test_outside_workspace_source_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as directory:
            folder=Path(directory)
            with self.assertRaisesRegex(ValueError,'Nonlocal/missing'):archive_files(folder,[Path(__file__)],folder/'bad.zip',{},reserve=0)

    def test_build_consumes_actual_supervisor_export_locations_and_rejects_bad_provenance(self):
        # Synthetic wiring fixture: lock/report verification is tested separately.
        # Actual archive and export-provenance validation run here.
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as directory:
            root=Path(directory);study=root/'outputs/researchforge-study';study.mkdir(parents=True)
            upstream=root/'work/NeuralCommonNeighbor';upstream.mkdir(parents=True)
            (upstream/'LICENSE').write_text('synthetic fixture only')
            lockpath=study/'lock.json';lockpath.write_text('{}')
            lock={'lock_sha256':'fixture-lock','payload':{'plan':{'datasets':{}},'artifacts':{}}}
            reports={}
            for dataset in STATUSES:
                path=study/'results/final'/dataset/'report.json';path.parent.mkdir(parents=True)
                report={'output_sha256':{}};path.write_text(json.dumps(report));reports[dataset]=(path,report)
            for folder,status in [('tables','Verified frozen-result table export'),('figures','Verified frozen-result figures')]:
                target=study/'results/final'/folder;target.mkdir(parents=True)
                payload=target/'negative-result.txt';payload.write_text('delta = -0.02\n')
                (target/'provenance.json').write_text(json.dumps({'status':status,'lock_sha256':'fixture-lock','files':{payload.name:digest(payload)}}))
            with patch('build_reproducibility_package.verify_lock',return_value=lock),patch('build_reproducibility_package.checked_report',side_effect=lambda r,l,d:reports[d]):
                dest=build(root,lockpath,study/'package.zip')
                names=verify_archive(dest)['files']
                for folder in ['tables','figures']:
                    self.assertIn(f'outputs/researchforge-study/results/final/{folder}/negative-result.txt',names)
                provenance=study/'results/final/figures/provenance.json'
                original=json.loads(provenance.read_text())
                altered=dict(original,status='incomplete');provenance.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,'Incomplete or wrong export'):build(root,lockpath,study/'bad-status.zip')
                altered=dict(original,files={'../tables/negative-result.txt':next(iter(original['files'].values()))});provenance.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,'outside its directory'):build(root,lockpath,study/'bad-path.zip')

if __name__=='__main__':unittest.main()
