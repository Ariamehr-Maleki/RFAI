"""Assemble a DRAFT from completed development artifacts, without test labels.

Never creates a usable final lock. Review development/runtime/sampling evidence,
then explicitly finalize the global plan with final_evaluation_gate.freeze_plan.
"""
import argparse
import ast
import json
import math
from pathlib import Path
from frozen_ranker import model_spec,digest
from final_evaluation_gate import validate_plan,has_completed_scores
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3];STUDY=ROOT/'outputs/researchforge-study';SCRIPTS=STUDY/'scripts'
RESULTS=STUDY/'results';BASE=RESULTS/'prescience-expanded-1200-1000-500-1000';SEEDS=[42,43,44]


def read(path):return json.loads(path.read_text())
def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def literal(name,variable):
    tree=ast.parse((SCRIPTS/name).read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==variable for t in node.targets):return ast.literal_eval(node.value)
    raise ValueError('Missing literal '+variable)


def require_completed_report(path):
    report=read(path);status=str(report.get('status','')).lower()
    if not status or any(word in status for word in ['running','incomplete','pending','failed','stopped','in progress']) or not has_completed_scores(report):
        raise ValueError('Incomplete development prerequisite: '+str(path))
    return report


def require_cost_report(path):
    """A file existing is insufficient: require every prespecified observation."""
    report=read(path)
    if report.get('status')!='Completed isolated development pipeline cost; not final quality evaluation':
        raise ValueError('Incomplete isolated cost prerequisite')
    for name,value in dict(query_count=30,repeats=3,model_seed=42,padding_seed=42,candidates=1000).items():
        if report.get(name)!=value:raise ValueError('Unexpected cost design: '+name)
    ids=report.get('query_ids',[])
    if len(ids)!=30 or len(set(ids))!=30:raise ValueError('Cost requires30 distinct development queries')
    if set(report.get('results',{}))!={'full','no_content'}:raise ValueError('Both cost pipelines required')
    expected={(qid,repeat) for qid in ids for repeat in range(3)}
    for mode,cell in report['results'].items():
        observations=cell.get('observations',[])
        if len(observations)!=90 or {(r.get('qid'),r.get('repeat')) for r in observations}!=expected:
            raise ValueError('Incomplete or duplicate cost observations: '+mode)
        for row in observations:
            timings=row.get('reranking_seconds',{})
            if set(timings)!={'lambdamart','binary_lightgbm','logistic'}:raise ValueError('Missing cost model')
            values=[row.get('retrieval_and_features_seconds'),*timings.values()]
            if any(type(v) not in [int,float] or not math.isfinite(v) or v<0 for v in values):raise ValueError('Invalid measured cost')
    return report


def assemble(count,reason,environment):
    if count<0 or len(reason.strip())<30:raise ValueError('Explicit census/sample count and development-based rationale required')
    required=[BASE/'report.json',BASE/'isolated_pipeline_cost.json',STUDY/'data-audit/final_semantic_index.json',
              STUDY/'data-audit/final_lexical_index.json',RESULTS/'ogb-expanded-10000/report.json',RESULTS/'ncn-expanded-10000/report.json',
              RESULTS/'exharmony-expanded-1000-500/report.json',Path(environment)]
    missing=[relative(p) for p in required if not p.is_file()]
    if missing:raise RuntimeError('Final-plan prerequisites incomplete: '+', '.join(missing))
    if not require_completed_report(BASE/'report.json')['status'].startswith('Completed expanded DEVELOPMENT'):raise ValueError('Expanded development incomplete')
    require_cost_report(BASE/'isolated_pipeline_cost.json')
    for path in [RESULTS/'ogb-expanded-10000/report.json',RESULTS/'ncn-expanded-10000/report.json',RESULTS/'exharmony-expanded-1000-500/report.json']:
        require_completed_report(path)
    for name in ['final_semantic_index.json','final_lexical_index.json']:
        if not read(STUDY/'data-audit'/name)['status'].startswith('PASS'):raise ValueError('Unverified final index: '+name)
    configs={}
    def config(name,metric,runner,policy):
        result=dict(primary_metric=metric,seeds=SEEDS,multiplicity='Holm',candidate_policy=policy,
            training_policy='Frozen saved development-selected models; no training or tuning on final labels.',
            uncertainty_policy='Average model-seed effects before paired cluster inference; marginal95% intervals and separate per-task Holm families.',
            test_split='Explicit locked held-out inputs; see dataset-specific fields.',runner_schema=1,inference=dict(repetitions=10000,seed=314159),
            output_directory=f'outputs/researchforge-study/results/final/{name}',methods={},artifacts=[])
        configs[name]=result
        for source in literal(runner,'CODE'):add(result,SCRIPTS/source,'code')
        add(result,STUDY/'PROTOCOL.md','protocol');add(result,STUDY/'SCIENTIFIC_REPORTS_PLAN.md','protocol');add(result,environment,'environment')
        return result
    def add(cfg,path,role):
        path=Path(path)
        if not path.is_file():raise FileNotFoundError(path)
        entry=dict(path=relative(path),role=role)
        old=next((a for a in cfg['artifacts'] if a['path']==entry['path']),None)
        if old and old!=entry:raise ValueError('Conflicting artifact roles')
        if not old:cfg['artifacts'].append(entry)
    def spec(cfg,folder,name):
        result=model_spec(folder,name)
        for key,role in [('path','model'),('source_report','development_report')]:
            add(cfg,result[key],role);result[key]=relative(result[key])
        return result
    p=config('prescience','ndcg1000','prescience_final_run.py','1000 candidates; full RRF and separately retrained graph-only pipeline; no held-out positive injection; padding42.')
    p.update(feature_names=literal('prescience_hybrid_graph.py','FEATURES'),candidates=1000,padding_seed=42,
        query_sampling=dict(count=count,seed=20261027),sampling_rationale=reason,
        corpus_inputs=dict(corpus='work/data/prescience/test.parquet',author_history='work/data/prescience/author_publications.jsonl'),
        index_inputs=dict(lexical='work/indices/prescience-test-hash18',semantic='work/indices/prescience-test-minilm'))
    add(p,BASE/'report.json','development_report');add(p,BASE/'query_manifest.json','split')
    # Bind the measured cost and preprocessing audits before any final labels.
    # Cost reports are evidence artifacts, not predictive-development scores.
    for evidence in [BASE/'isolated_pipeline_cost.json',STUDY/'data-audit/final_semantic_index.json',STUDY/'data-audit/final_lexical_index.json',
                     STUDY/'data-audit/train_semantic_index.json',STUDY/'data-audit/train_encoding_cost.json',STUDY/'data-audit/loader-equivalence.json']:
        add(p,evidence,'data')
    for value in p['corpus_inputs'].values():add(p,ROOT/value,'data')
    for mode in ['full','no_content']:
        for family in ['lambdamart','binary_lightgbm','logistic']:
            p['methods'][mode+'_'+family]=dict(kind='learned',candidate_mode=mode,models={str(s):spec(p,BASE/mode/f'seed-{s}',family) for s in SEEDS})
    for name in ['frequency10','frequency45','frequency_all','common_neighbors10','autoresearch_description','rrf','lexical_mean10','semantic_mean10','category_cosine']:
        p['methods'][name]=dict(kind='feature',candidate_mode='full',feature=name)
    for name in ['frequency10_standalone','autoresearch_description_standalone','lexical_standalone','semantic_standalone','rrf_standalone']:
        p['methods'][name]=dict(kind='standalone',candidate_mode='full',ranking=name)
    # Complete eight-cell factorial, preserving each feature schema and seed.
    cells=read(BASE/'full/seed-42/factorial-s42/report.json')['cells']
    for name in cells:p['methods']['factorial_'+name]=dict(kind='learned',candidate_mode='full',models={str(s):spec(p,BASE/'full'/f'seed-{s}'/f'factorial-s{s}',name) for s in SEEDS})
    sensitivity=BASE/'full/seed-42/fixed-scorer-sensitivity/report.json';report=read(sensitivity);variants=list(report['metrics'])
    chosen=max(variants,key=lambda n:report['metrics'][n]['shared']['ndcg1000'])
    p['methods']['fixed_selected']=dict(kind='fixed',candidate_mode='full',variant=chosen)
    p['fixed_selection']=dict(criterion='Expanded seed42 shared validation nDCG1000; first report variant wins exact ties.',variant=chosen,source_report=relative(sensitivity));add(p,sensitivity,'development_report')
    full='full_lambdamart';factorial='factorial_text+network+explicit_time'
    p['primary_comparisons']=[[full,b] for b in ['full_binary_lightgbm','frequency45','semantic_standalone','rrf','fixed_selected']]+[[factorial,'factorial_'+b] for b in ['network+explicit_time','text+explicit_time','text+network']]
    p['subgroup_comparisons']=[[full,b] for b in ['full_binary_lightgbm','no_content_lambdamart']]+p['primary_comparisons'][-3:]
    for key,value in p['index_inputs'].items():
        folder=ROOT/value;add(p,folder/'manifest.json','data')
        files=read(folder/'manifest.json')['files'] if key=='lexical' else ['embeddings.f16']
        for name in files:add(p,folder/name,'data')
    o=config('ogbl-collab','hits50','ogb_final_run.py','Official positive edges and shared100000 negatives; training graph through2017; no2018 edges.')
    add(o,STUDY/'ncn-runtime.json','environment')
    raw=ROOT/'work/data/ogbl-collab/extracted/collab/raw'
    o.update(graph_cutoff_year=2017,negative_count=100000,batch_size=4096,
        inputs={name:relative(raw/file) for name,file in dict(edges='edge.csv.gz',years='edge_year.csv.gz',weights='edge_weight.csv.gz',nodes='node-feat.csv.gz').items()})
    o['inputs']['split']=relative(raw.parent/'split/time/test.pt')
    for path in o['inputs'].values():add(o,ROOT/path,'data')
    add(o,RESULTS/'ogb-expanded-10000/config.json','split');add(o,RESULTS/'ogb-expanded-10000/report.json','development_report')
    add(o,RESULTS/'ncn-expanded-10000/report.json','development_report')
    for track in ['graph','graph_plus_released_nodes']:
        source_track=track
        for family in ['lambdamart','binary_lightgbm']:
            files={str(s):RESULTS/'ogb-expanded-10000'/f's{s}-{source_track}__{family}.txt' for s in SEEDS}
            for path in files.values():add(o,path,'model')
            o['methods'][source_track+'__'+family]=dict(kind='tree',track=track,models={s:relative(path) for s,path in files.items()})
    for feature in literal('ogb_validation.py','NAMES'):o['methods'][feature]=dict(kind='feature',feature=feature)
    o['methods']['released_node_cosine']=dict(kind='feature',feature='released_node_cosine')
    ncn={str(s):RESULTS/f'ncn-expanded-10000-e100-s{s}/best.pt' for s in SEEDS}
    for path in ncn.values():add(o,path,'model');add(o,path.parent/'report.json','development_report')
    o['methods']['ncn']=dict(kind='ncn',models={s:relative(path) for s,path in ncn.items()})
    for path in [SCRIPTS/'ncn_temporal.py',ROOT/'work/NeuralCommonNeighbor/model.py',ROOT/'work/NeuralCommonNeighbor/utils.py']:add(o,path,'code')
    o['primary_comparisons']=[['graph__lambdamart',b] for b in ['graph__binary_lightgbm','adamic_adar','ncn']]
    e=config('exHarmony','ndcg10','exharmony_final_run.py','Shared1000-candidate hashed BM25 author pool; no positive injection; all1589 adapted held-out query IDs.')
    e.update(feature_names=literal('exharmony_bm25_pilot.py','FEATURES'),candidates=1000,primary_judgment='authors',query_sampling=dict(count=0,seed=42),
        index_directory='work/indices/exharmony-hash18',inputs=dict(queries='work/data/exHarmony/papers_test.jsonl',judgeable='work/data/exHarmony/papers_test_judgeable.jsonl',split='outputs/researchforge-study/data-audit/exharmony_split.json',
            **{j:f'work/data/exHarmony/qrels/qrels.test.{j}.tsv' for j in ['authors','cite','simcite']}))
    for key,path in e['inputs'].items():add(e,ROOT/path,'split' if key=='split' else 'data')
    for name in literal('exharmony_final_run.py','INDEX_FILES'):add(e,ROOT/e['index_directory']/name,'data')
    for family in ['lambdamart','binary_lightgbm']:
        models={}
        for seed in SEEDS:
            folder=RESULTS/f'exharmony-bm25-1000-500-1000-s{seed}';path=folder/(family+'.txt');source=folder/'report.json'
            add(e,path,'model');add(e,source,'development_report')
            models[str(seed)]=dict(kind='lightgbm',features=e['feature_names'],path=relative(path),sha256=digest(path),source_report=relative(source),source_report_sha256=digest(source))
        e['methods'][family]=dict(kind='learned',models=models)
    for name in e['feature_names']:e['methods'][name]=dict(kind='feature',feature=name)
    e['primary_comparisons']=[['lambdamart',b] for b in ['binary_lightgbm','max_paper_bm25']]
    plan=dict(schema_version=1,stage='ready_for_final_test',datasets=configs)
    validate_plan(plan)
    plan['stage']='draft_pending_development_review'
    plan['review_required']='Verify all chosen models, development results, sampling rationale and runtime inventory; this draft cannot authorize test access.'
    return plan


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prescience-queries',required=True,type=int);ap.add_argument('--sampling-rationale',required=True)
    ap.add_argument('--environment',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):raise ValueError('New workspace output required')
    atomic_state(args.output,assemble(args.prescience_queries,args.sampling_rationale,args.environment))
