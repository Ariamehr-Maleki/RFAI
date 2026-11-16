"""Fail-closed configuration/artifact lock for eventual held-out evaluation.

No real configuration is finalized by importing this module. It never parses
benchmark labels. Future final runners must call authorize_evaluation BEFORE
opening test labels, and must consume the returned frozen dataset configuration.
This is a reproducibility/audit guard, not an OS access-control boundary.
"""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

DATASETS={'prescience':'ndcg1000','ogbl-collab':'hits50','exHarmony':'ndcg10'}
ROLES={'code','model','data','split','protocol','environment','development_report'}


def has_completed_scores(report):
    """Recognize aggregate reports or a complete, audited NCN epoch history."""
    if not isinstance(report,dict):return False
    if any(report.get(k) for k in ['metrics','seed_summary','per_seed']):return True
    # NCN's actual report schema stores best_validation_hits50 and history,
    # rather than a metrics dictionary. Require every declared epoch and exact
    # consistency of the selected score; never accept a partial checkpoint log.
    history=report.get('history');epochs=report.get('config',{}).get('epochs')
    if not isinstance(history,list) or type(epochs) is not int or epochs<1 or len(history)!=epochs:return False
    if [row.get('epoch') for row in history]!=list(range(1,epochs+1)):return False
    values=[row.get('validation_hits50') for row in history];best=report.get('best_validation_hits50')
    if any(type(v) not in [int,float] or not math.isfinite(v) or not 0<=v<=1 for v in [best,*values]):return False
    return abs(best-max(values))<=1e-12


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')


def digest_file(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def local_file(root,relative):
    root=Path(root).resolve();relative=Path(relative)
    if relative.is_absolute():raise ValueError('Artifact paths must be workspace-relative')
    result=(root/relative).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError(f'Invalid or missing artifact: {relative}')
    return result


def validate_plan(plan):
    if plan.get('schema_version')!=1 or plan.get('stage')!='ready_for_final_test':
        raise ValueError('Development/draft configurations cannot authorize test access')
    if set(plan.get('datasets',{}))!=set(DATASETS):
        raise ValueError('Freeze all three dataset configurations before any test run')
    for dataset,metric in DATASETS.items():
        config=plan['datasets'][dataset]
        if config.get('primary_metric')!=metric:raise ValueError(f'Unexpected primary metric: {dataset}')
        seeds=config.get('seeds',[])
        if len(set(seeds))<3 or not all(isinstance(s,int) for s in seeds):
            raise ValueError('At least three distinct integer seeds are required')
        methods=config.get('methods',{})
        if len(methods)<2 or any(not isinstance(v,dict) or not v for v in methods.values()):
            raise ValueError('Explicit per-method configurations are required')
        for key in ['candidate_policy','training_policy','test_split','uncertainty_policy']:
            if not isinstance(config.get(key),str) or not config[key].strip():raise ValueError(f'Missing {dataset}/{key}')
        comparisons=config.get('primary_comparisons',[])
        if not comparisons or any(len(pair)!=2 or pair[0]==pair[1] or any(m not in methods for m in pair) for pair in comparisons):
            raise ValueError('Declare actual method pairs before inspecting test results')
        if config.get('multiplicity')!='Holm':raise ValueError('Primary family requires Holm correction')
        artifacts=config.get('artifacts',[])
        if not ROLES.issubset({a.get('role') for a in artifacts}):
            raise ValueError(f'Missing artifact roles for {dataset}')
        if len({a.get('path') for a in artifacts})!=len(artifacts):
            raise ValueError('Duplicate artifact paths within a dataset')


def freeze_plan(root,plan,destination):
    """Write an immutable new lock only after every referenced artifact exists."""
    root=Path(root).resolve();destination=Path(destination).resolve()
    if not destination.is_relative_to(root):raise ValueError('Lock must remain in task workspace')
    validate_plan(plan)
    artifacts={}
    for config in plan['datasets'].values():
        for entry in config['artifacts']:
            path=local_file(root,entry['path'])
            relative=path.relative_to(root).as_posix()
            if entry['role']=='development_report':
                report=json.loads(path.read_text(encoding='utf-8'))
                status=str(report.get('status','')).lower() if isinstance(report,dict) else ''
                has_scores=has_completed_scores(report)
                if not status or any(word in status for word in ['running','incomplete','pending','failed','stopped','in progress']) or not has_scores:
                    raise ValueError('Development report is not a completed JSON report')
            artifacts[relative]=dict(bytes=path.stat().st_size,sha256=digest_file(path))
    payload=dict(schema_version=1,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 plan=plan,artifacts=artifacts)
    lock=dict(payload=payload,lock_sha256=hashlib.sha256(canonical(payload)).hexdigest())
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as f:json.dump(lock,f,indent=2,ensure_ascii=False)
    return lock['lock_sha256']


def verify_lock(root,lock_path):
    root=Path(root).resolve();path=Path(lock_path).resolve()
    if not path.is_relative_to(root):raise ValueError('Lock outside workspace')
    lock=json.loads(path.read_text(encoding='utf-8'));payload=lock['payload']
    if hashlib.sha256(canonical(payload)).hexdigest()!=lock['lock_sha256']:
        raise ValueError('Lock content has changed')
    validate_plan(payload['plan'])
    required={(root/entry['path']).resolve().relative_to(root).as_posix()
              for cfg in payload['plan']['datasets'].values() for entry in cfg['artifacts']}
    if required!=set(payload['artifacts']):raise ValueError('Artifact inventory does not match configuration')
    for relative,record in payload['artifacts'].items():
        target=local_file(root,relative)
        if target.stat().st_size!=record['bytes'] or digest_file(target)!=record['sha256']:
            raise ValueError(f'Locked artifact changed: {relative}')
    return lock


def authorize_evaluation(root,lock_path,dataset,run_id,ledger_dir):
    """One recorded run per dataset; identical run ID can resume after a crash."""
    root=Path(root).resolve();ledger_dir=Path(ledger_dir).resolve()
    if dataset not in DATASETS or not run_id or not isinstance(run_id,str):raise ValueError('Invalid dataset/run')
    if not ledger_dir.is_relative_to(root):raise ValueError('Ledger outside workspace')
    lock=verify_lock(root,lock_path)
    ledger_dir.mkdir(parents=True,exist_ok=True);path=ledger_dir/(dataset+'.json')
    identity=dict(dataset=dataset,run_id=run_id,lock_sha256=lock['lock_sha256'])
    try:
        with path.open('x',encoding='utf-8') as f:
            json.dump(dict(**identity,first_authorized_utc=dt.datetime.now(dt.timezone.utc).isoformat()),f,indent=2)
    except FileExistsError:
        existing=json.loads(path.read_text(encoding='utf-8'))
        if any(existing.get(k)!=v for k,v in identity.items()):
            raise ValueError('A different held-out evaluation run/lock is already recorded')
    return lock['payload']['plan']['datasets'][dataset]
