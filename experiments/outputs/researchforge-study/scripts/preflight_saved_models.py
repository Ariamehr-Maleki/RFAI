"""Read-only development artifact checks; never opens held-out labels.

LightGBM uses three synthetic rows for a serialization/interface probe only.
NCN checkpoints are inventoried, not deserialized or executed in this audit.
"""
import datetime as dt
import json
from pathlib import Path
import numpy as np
from lightgbm import Booster
from frozen_ranker import digest
from supervisor_state import atomic_state

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'outputs/researchforge-study'
RESULTS = STUDY / 'results'


def read(path):
    return json.loads(path.read_text())


def probe(path, width, family, selected):
    model = Booster(model_file=str(path))
    if model.num_feature() != width:
        raise ValueError('Feature width differs: ' + str(path))
    objective = model.dump_model()['objective'].split()[0]
    if objective != ('lambdarank' if family == 'lambdamart' else 'binary'):
        raise ValueError('Objective differs: ' + str(path))
    if int(model.params['num_leaves']) != selected['leaves'] or model.num_trees() != selected['trees']:
        raise ValueError('Saved model differs from development selection: ' + str(path))
    x = np.stack([np.zeros(width), np.ones(width), np.linspace(0, 1, width)])
    values = model.predict(x, num_threads=1)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('Nonfinite synthetic inference: ' + str(path))
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path),
                width=width, objective=objective, trees=model.num_trees(),
                selected_leaves=selected['leaves'], synthetic_probe='PASS; not a quality evaluation')


def main():
    report = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  status='Running', lightgbm=[], ncn=[], source_sha256={},
                  limitations=['No held-out labels opened.',
                               'New hashes establish the current inventory, not historical checkpoint integrity.',
                               'NCN checkpoint loading and inference are not exercised by this lightweight audit.',
                               'Expanded PreScience models and global freeze remain pending.'])
    destination = STUDY / 'data-audit/final-model-preflight.json'
    try:
        op = RESULTS / 'ogb-expanded-10000/report.json'
        ogb = read(op)
        report['source_sha256'][op.relative_to(ROOT).as_posix()] = digest(op)
        for seed in [42, 43, 44]:
            for track, width in [('graph', 8), ('graph_plus_released_nodes', 266)]:
                for family in ['lambdamart', 'binary_lightgbm']:
                    name = track + '__' + family
                    selected = ogb['runs'][str(seed)]['methods'][name]
                    report['lightgbm'].append(probe(op.parent / f's{seed}-{name}.txt', width, family, selected))
            folder = RESULTS / f'exharmony-bm25-1000-500-1000-s{seed}'
            ep = folder / 'report.json'
            exh = read(ep)
            report['source_sha256'][ep.relative_to(ROOT).as_posix()] = digest(ep)
            if len(exh['features']) != 7 or len(set(exh['features'])) != 7:
                raise ValueError('Invalid exHarmony ordered feature schema')
            for family in ['lambdamart', 'binary_lightgbm']:
                selected = max([c for c in exh['tuning'] if c['method'] == family], key=lambda c: c['ndcg10'])
                report['lightgbm'].append(probe(folder / (family + '.txt'), 7, family, selected))
            folder = RESULTS / f'ncn-expanded-10000-e100-s{seed}'
            npth = folder / 'report.json'
            ncn = read(npth)
            verification = read(folder / 'independent_metric_verification.json')
            if len(ncn['history']) != 100 or verification['status'] != 'PASS' or verification['completed_epochs'] != 100:
                raise ValueError('Incomplete NCN development run')
            for filename, expected in verification['sha256'].items():
                if digest(folder / filename) != expected:
                    raise ValueError('NCN independently verified artifact changed: ' + filename)
            for filename, key in [('model.py', 'model_sha256'), ('utils.py', 'utils_sha256')]:
                if digest(ROOT / 'work/NeuralCommonNeighbor' / filename) != ncn['config'][key]:
                    raise ValueError('NCN upstream source changed')
            checkpoint = folder / 'best.pt'
            if checkpoint.stat().st_size == 0:
                raise ValueError('Empty NCN checkpoint')
            report['source_sha256'][npth.relative_to(ROOT).as_posix()] = digest(npth)
            report['ncn'].append(dict(seed=seed, checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                sha256=digest(checkpoint), bytes=checkpoint.stat().st_size, completed_epochs=100,
                selected_epoch=verification['first_best_epoch'], development_metric_verification='PASS'))
        report['status'] = 'PASS for18 saved LightGBM interfaces and3 NCN artifact inventories; full study pending'
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = repr(error)
        raise
    finally:
        report['audit_script_sha256'] = digest(Path(__file__))
        atomic_state(destination, report)
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
