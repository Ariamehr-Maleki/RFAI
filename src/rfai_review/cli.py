"""Read-only reviewer commands; experiments are never launched by import."""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def audit(root=ROOT):
    root=Path(root).resolve();manifest=load(root/'MANIFEST.json')['files']
    for rel,info in manifest.items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root):raise ValueError('Manifest path escapes repository: '+rel)
        if not p.is_file():raise ValueError('Missing payload: '+rel)
        if p.stat().st_size!=info['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError('Changed payload: '+rel)
    archived=load(root/'provenance/archived-source-inventory.json')
    for rel,info in archived.items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError('Archived experiment source changed: '+rel)
    reports={k:load(root/'results'/f'{stem}-report.json') for k,stem in [('prescience','prescience'),('ogbl-collab','ogb'),('exHarmony','exharmony')]}
    for k,r in reports.items():
        if not r['status'].startswith('Completed frozen'):raise ValueError('Nonfinal report: '+k)
    checked=0
    with (root/'results/benchmark_results.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            d,m,k=row['dataset'],row['method'],row['metric']
            if d=='prescience':v=reports[d]['seed_summary'][m][k]
            elif d=='ogbl-collab':v=reports[d]['seed_summary'][m]
            else:v=reports[d]['seed_summary'][row['judgment']][m][k]
            if float(row['mean'])!=v['mean'] or float(row['sample_sd'])!=v['sample_sd']:
                raise ValueError('Numerical export mismatch: '+str(row))
            checked+=1
    if checked!=568:raise ValueError('Unexpected numerical inventory length')
    return {'status':'PASS compact review audit','payloads_verified':len(manifest),'archived_sources_verified':len(archived),'numerical_rows_verified':checked,'scope':'File identity and aggregate numerical consistency; not full prediction replay.'}
def summary(root=ROOT):
    r=Path(root)/'results';p=load(r/'prescience-report.json');o=load(r/'ogb-report.json');e=load(r/'exharmony-report.json');u=load(r/'official-comparator-results.json')
    return {'PreScience nDCG@1000':{'ranker':p['seed_summary']['full_lambdamart']['ndcg1000']['mean'],'matched_classifier':p['seed_summary']['full_binary_lightgbm']['ndcg1000']['mean'],'official_harmonized':u['results']['harmonized_historical_input']['metrics']['ndcg1000']},'ogbl-collab Hits@50':{k:o['seed_summary'][k]['mean'] for k in ['graph__lambdamart','graph__binary_lightgbm','ncn']},'adapted exHarmony Authors nDCG@10':{k:e['seed_summary']['authors'][k]['ndcg10']['mean'] for k in ['lambdamart','binary_lightgbm','max_paper_bm25']},'Additional PreScience post-result check':load(r/'additional-cohort/results.json'), 'interpretation':'Conditional gains only; no universal superiority or eight-agent validation.'}
def fixtures(root=ROOT):
    folder=Path(root)/'experiments/outputs/researchforge-study/scripts'
    names=['test_final_statistics.py','test_final_evaluation_gate.py','test_protocol.py','test_hashed_bm25.py','test_frozen_inference.py']
    for name in names:
        code=subprocess.call([sys.executable,'-m','unittest','discover','-s',str(folder),'-p',name,'-v'],cwd=root)
        if code:return code
    extra=Path(root)/'experiments/outputs/researchforge-study/review-20261107'
    for name in ['test_official_comparator.py','test_indexed_helper.py','test_bounded_helper.py']:
        code=subprocess.call([sys.executable,'-m','unittest','discover','-s',str(extra),'-p',name,'-v'],cwd=root)
        if code:return code
    return 0
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['audit','results','fixtures']);args=ap.parse_args(argv)
    try:
        if args.command=='fixtures':return fixtures()
        print(json.dumps(audit() if args.command=='audit' else summary(),indent=2));return 0
    except (ValueError,KeyError,FileNotFoundError) as exc:
        print('FAIL: '+str(exc),file=sys.stderr);return 1
