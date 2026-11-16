"""Read-only inference adapters for saved development models.

No training or benchmark-label reader. The future final runner must authorize
test access separately and include this module, model and schema in its lock.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def model_spec(folder,name):
    """Export the selected artifact and ordered features, never select a model."""
    folder=Path(folder).resolve()
    report=json.loads((folder/'report.json').read_text())
    if not str(report.get('status','')).lower().startswith('development'):
        raise ValueError('Completed development report required')
    if name in report.get('models',{}):
        cell=report['models'][name];features=cell['feature_names'];hashes=cell['artifacts']
        kind='logistic_pipeline' if name=='logistic' else 'lightgbm'
    elif name in report.get('cells',{}):
        cell=report['cells'][name];features=cell['features'];hashes=cell['artifact_sha256'];kind='lightgbm'
    else:raise ValueError('No saved learned model for '+name)
    suffix='.joblib' if kind=='logistic_pipeline' else '.txt'
    path=folder/(name+suffix)
    if path.name not in hashes or digest(path)!=hashes[path.name]:
        raise ValueError('Selected model artifact changed')
    if not features or len(features)!=len(set(features)):raise ValueError('Invalid feature schema')
    return dict(name=name,kind=kind,path=str(path),sha256=hashes[path.name],features=features,
                source_report=str(folder/'report.json'),source_report_sha256=digest(folder/'report.json'))


class FrozenRanker:
    def __init__(self,spec,input_features):
        self.spec=dict(spec);self.input_features=tuple(input_features)
        if len(input_features)!=len(set(input_features)) or not set(spec['features'])<=set(input_features):
            raise ValueError('Missing or duplicate input features')
        if digest(spec['path'])!=spec['sha256']:raise ValueError('Frozen model changed')
        self.columns=[input_features.index(name) for name in spec['features']]
        if spec['kind']=='lightgbm':
            from lightgbm import Booster
            self.model=Booster(model_file=spec['path'])
            width=self.model.num_feature()
        elif spec['kind']=='logistic_pipeline':
            import joblib
            # Only locally generated, SHA-verified model artifacts are supported.
            self.model=joblib.load(spec['path']);width=self.model.n_features_in_
        else:raise ValueError('Unsupported frozen model kind')
        if width!=len(self.columns):raise ValueError('Model width and feature schema differ')

    def predict(self,matrix):
        matrix=np.asarray(matrix)
        if matrix.ndim!=2 or matrix.shape[1]!=len(self.input_features) or not np.isfinite(matrix).all():
            raise ValueError('Finite feature matrix with exact declared width required')
        selected=matrix[:,self.columns]
        if self.spec['kind']=='lightgbm':values=self.model.predict(selected,num_threads=1)
        else:values=self.model.predict_proba(selected)[:,1]
        if values.shape!=(len(matrix),) or not np.isfinite(values).all():raise ValueError('Invalid inference output')
        return values
