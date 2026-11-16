"""Fresh-process isolated NCN integration, temporary synthetic fixtures only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'work/ncn-runtime'))
sys.path.insert(1,str(ROOT/'work/NeuralCommonNeighbor'))
import unittest
import numpy as np
import pandas as pd
import torch
from torch_sparse import SparseTensor
from test_ogb_final_run import OGBFinalTests
from final_evaluation_gate import freeze_plan
from ogb_final_run import run
from ncn_temporal import make_models


class NCNFinalTest(unittest.TestCase):
    def test_saved_model_inference_matches_direct_unbatched_reference(self):
        self.assertEqual(torch.__version__,'2.5.1+cpu');torch.set_num_threads(1)
        fixture=OGBFinalTests();fixture.setUp()
        try:
            cfg=fixture.plan['datasets']['ogbl-collab'];cfg['batch_size']=32
            nodes=pd.read_csv(ROOT/cfg['inputs']['nodes'],header=None,dtype=np.float32).to_numpy()
            edges=pd.read_csv(ROOT/cfg['inputs']['edges'],header=None).to_numpy(dtype=np.int64)
            split=torch.load(ROOT/cfg['inputs']['split'],weights_only=False)
            pairs=torch.cat([split['edge'],split['edge_neg']],0).T.contiguous()
            torch.manual_seed(42);model,predictor=make_models(torch);model.eval();predictor.eval()
            checkpoint=fixture.folder/'synthetic-ncn.pt';torch.save(dict(model=model.state_dict(),predictor=predictor.state_dict()),checkpoint)
            adj=SparseTensor.from_edge_index(torch.from_numpy(edges.T.copy()),sparse_sizes=(len(nodes),len(nodes))).to_symmetric().coalesce()
            with torch.inference_mode():expected=predictor(model(torch.from_numpy(nodes),adj),adj,pairs).flatten().numpy()
            cfg['methods']['ncn']=dict(kind='ncn',models={str(s):fixture.relative(checkpoint) for s in cfg['seeds']})
            cfg['artifacts'].append(dict(role='model',path=fixture.relative(checkpoint)))
            cfg['artifacts'] += [dict(role='code',path=fixture.relative(p)) for p in [Path(__file__).with_name('ncn_temporal.py'),ROOT/'work/NeuralCommonNeighbor/model.py',ROOT/'work/NeuralCommonNeighbor/utils.py']]
            freeze_plan(ROOT,fixture.plan,fixture.lock);report=run(ROOT,fixture.lock,'fixture',fixture.ledger)
            self.assertIn('ncn',report['per_seed'])
            for seed in cfg['seeds']:
                with np.load(fixture.folder/'result'/f'ncn.s{seed}.predictions.npz') as saved:
                    actual=np.concatenate([saved['positive'],saved['negative']])
                np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
        finally:fixture.tearDown()


if __name__=='__main__':unittest.main(defaultTest='NCNFinalTest')
