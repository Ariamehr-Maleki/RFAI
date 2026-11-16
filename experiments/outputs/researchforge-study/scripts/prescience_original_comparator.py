"""Score saved DEVELOPMENT candidate pools with disclosed original-score adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from prescience_pilot import load_data,evaluate
from researchforge_scoring import history_adapter,ADAPTATION


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');args=ap.parse_args()
    run=Path(args.run); start=time.perf_counter()
    assert json.loads((run/'report.json').read_text())['status'].startswith('Development')
    folder=run/'original-score-adapter';folder.mkdir(exist_ok=True)
    papers,histories,dates=load_data()
    meta=[json.loads(p.read_text()) for p in sorted((run/'valid_queries').glob('*.json'))]
    assert meta
    scores=[]
    for q in meta:
        scores.extend(history_adapter(q['seed'],q['candidates'],q['date'],papers,histories,dates))
    groups=np.array([len(q['candidates']) for q in meta])
    agg,rows=evaluate(np.array(scores),groups,meta,True)
    (folder/'researchforge_history_categories.predictions.json').write_text(json.dumps(rows))
    root=Path(__file__).resolve().parents[3]
    report=dict(status='Development shared-pool data-adapted original scorer; not original end-to-end reproduction',
                metrics=agg,adaptation=ADAPTATION,queries=len(meta),wall_seconds=time.perf_counter()-start,
                original_source_sha256=hashlib.sha256((root/'work/ResearchForge-AI/agent.py').read_bytes()).hexdigest(),
                adapter_sha256=hashlib.sha256(Path(__file__).with_name('researchforge_scoring.py').read_bytes()).hexdigest())
    (folder/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
