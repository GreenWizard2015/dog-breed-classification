# script to generate plots for ablation study
import argparse, os, sys
# add parent folder to path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

from Utils.WandBUtils import CWBProject
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def findHSLLayer(model):
  hsl = [
    layer 
    for layer in model.layers
    if layer.__class__.__name__ == 'CHyperSoftmaxLayer'
  ]
  assert 1 == len(hsl), 'There should be exactly one CHyperSoftmaxLayer'
  return hsl[0]

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument(
    '--wandb-project', type=str, help='Wandb project name (entity/project)',
    default='green_wizard/dog-breed-classification'
  )
  args = parser.parse_args()
  WBProject = CWBProject(args.wandb_project)
  allRuns = WBProject.groups(onlyBest=True)

  # plot mapping for 64d case, with different number of views
  NSamples = 10000
  mappingX = tf.linspace(-1.0, 1.0, NSamples).numpy()
  runsNames = [
    'HS-64 (base), 1 views, 224x224, RGB',
    'HS-64 (base), 2 views, 224x224, RGB',
    'HS-64 (base), 4 views, 224x224, RGB',
    'HS-64 (base), 6 views, 224x224, RGB',
  ]
  mappings = {}
  for runName in runsNames:
    assert runName in allRuns, f'Run {runName} not found'
    run = allRuns[runName]
    model = run.instantiateModel()
    
    hsl = findHSLLayer(model)
    mapping = hsl.mapping(mappingX).numpy()      
    mappings[runName] = mapping
    continue
  # verify that all mappings are increasing
  for runName, mapping in mappings.items():
    assert np.all(0.0 <= np.diff(mapping)), f'Mapping for {runName} is not increasing'
    continue
  # plot mappings with labels
  plt.figure(figsize=(10, 10))
  for runName, mapping in mappings.items():
    # label is "? views"
    label = runName.split(',')[1].strip()
    plt.plot(mappingX, mapping, label=label)
    continue
  # save plot
  plt.title('Mapping for 64d case, with different number of views')
  plt.legend()
  plt.tight_layout()
  plt.savefig('mapping-64d-per-views.png')
  plt.close()
  ########################
  # plot mapping for different dimensions, with 1 views
  runsNames = [
    'HS-64 (base), 1 views, 224x224, RGB',
    'HS-32 (base), 1 views, 224x224, RGB',
    'HS-16 (base), 1 views, 224x224, RGB',
    'HS-8 (base), 1 views, 224x224, RGB',
    'HS-4 (base), 1 views, 224x224, RGB',
    'HS-3 (base), 1 views, 224x224, RGB',
    'HS-2 (base), 1 views, 224x224, RGB',
  ]
  mappings = {}
  for runName in runsNames:
    assert runName in allRuns, f'Run {runName} not found'
    run = allRuns[runName]
    model = run.instantiateModel()
    
    hsl = findHSLLayer(model)
    mapping = hsl.mapping(mappingX).numpy()      
    mappings[runName] = mapping
    continue
  # verify that all mappings are increasing
  for runName, mapping in mappings.items():
    assert np.all(0.0 <= np.diff(mapping)), f'Mapping for {runName} is not increasing'
    continue
  # plot mappings with labels
  plt.figure(figsize=(10, 10))
  for runName, mapping in mappings.items():
    # label is number of dimensions, extracted from run config
    run = allRuns[runName]
    label = '%dd' % (run.config['classVecSize'],)
    plt.plot(mappingX, mapping, label=label)
    continue
  # save plot
  plt.title('Mapping for different dimensions, 1 view')
  plt.legend()
  plt.tight_layout()
  plt.savefig('mapping-per-dimensions.png')
  plt.close()