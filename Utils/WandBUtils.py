import wandb
import tempfile
import yaml
from functools import lru_cache
from NN.model import model_from_config

def _fixWandbConfig(config):
  # if config is a dict with 'desc' and 'value' keys, return value
  if isinstance(config, dict):
    if 'desc' in config and 'value' in config:
      return _fixWandbConfig(config['value'])

    return {k: _fixWandbConfig(v) for k,v in config.items()}
  return config

class CWBRun:
  def __init__(self, runId, api=None, tmpFolder=None):
    self._runId = runId
    self._api = api or wandb.Api()
    self._run = self._api.run(runId)
    self._tmpFolder = tmpFolder or tempfile.gettempdir()
    return
  
  @property
  @lru_cache(maxsize=1)
  def config(self):
    # load "config.yaml" from files of the run and return it as dict
    config = self._run.file('config.yaml')
    with config.download(self._tmpFolder, replace=True, exist_ok=True) as f:
      res = yaml.safe_load(f)
    return _fixWandbConfig(res)
  
  def models(self):
    # return list of models in the run
    res = []
    for raw in self._run.logged_artifacts():
      artifact = CWBFileArtifact(raw, self._tmpFolder, self._run)
      if artifact.name.lower().endswith('.h5'):
        res.append(artifact)
      continue

    return res
  
  @property
  def bestModel(self):
    # find 'model-latest.h5' in the run
    models = self.models()
    return next(f for f in models if f.name == 'model-latest.h5')
    
  @property
  @lru_cache(maxsize=1)
  def bestLoss(self):
    try:
      return min([x['val_loss'] for x in self.history()])
    except:
      return float('inf') # no history
    return
  
  @lru_cache(maxsize=1)
  def history(self):
    return self._run.scan_history()
  
  @property
  def name(self): return self._run.name

  @property
  def id(self): return self._run.id

  @property
  def fullId(self): return self._runId

  def instantiateModel(self):
    model = model_from_config(self.config)
    model.load_weights(self.bestModel.pathTo())
    return model
# End of CWBRun

class CWBFileArtifact:
  def __init__(self, artifact, tmpFolder, run):
    self._artifact = artifact
    self._tmpFolder = tmpFolder
    self._run = run
    return
  
  def pathTo(self):
    file = self._run.use_artifact(self._artifact)
    return file.file(self._tmpFolder)
  
  @property
  def name(self):
    res = self._artifact.name
    # format: "run-{id}-{name}:{version}"
    # we need only name
    res = res.split(':')[0] # remove version
    res = res.split('-')[2:] # remove "run" and id
    return '-'.join(res)
# End of CWBFileArtifact

class CWBProject:
  def __init__(self, projectId, api=None, tmpFolder=None):
    self._projectId = projectId
    self._api = api or wandb.Api()
    self._tmpFolder = tmpFolder or tempfile.gettempdir()
    return
  
  def runs(self, filters=None):
    runs = self._api.runs(self._projectId, filters=filters)
    return [CWBRun(self._projectId + '/' + run.id, self._api, self._tmpFolder) for run in runs]
  
  def groups(self, onlyBest=False):
    runs = self.runs()
    # group runs by name
    groups = {}
    for run in runs:
      name = run.name
      if name not in groups: groups[name] = []
      groups[name].append(run)
      continue

    if onlyBest: # select only best runs
      groups = {k: min(v, key=lambda x: x.bestLoss) for k, v in groups.items()}
      
    return groups
# End of CWBProject