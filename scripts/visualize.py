import argparse, os, sys
# add parent folder to path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

from NN.model import model_from_config
from Utils.WandBUtils import CWBRun
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--wandb-run', type=str, help='Wandb run full id (entity/project/run_id)', required=True)
  
  args = parser.parse_args()
  run = CWBRun(args.wandb_run)
  modelConfig = run.config
  model = model_from_config(modelConfig)
  model.summary()
  model.load_weights(run.bestModel.pathTo())
  # find layer with class CHyperSoftmaxLayer
  hsl = [
    layer 
    for layer in model.layers
    if layer.__class__.__name__ == 'CHyperSoftmaxLayer'
  ]
  assert 1 == len(hsl), 'There should be exactly one CHyperSoftmaxLayer'
  hsl = hsl[0]
  
  x = tf.linspace(-1.0, 1.0, 10000)
  mapping = hsl.mapping(x).numpy()
  # plot mapping
  import matplotlib.pyplot as plt
  plt.plot(x, mapping)
  # save plot
  plt.savefig('mapping.png')
  plt.close()

  # process classes/embeddings
  classes = hsl.embeddings.numpy()
  projected = classes
  if not(2 == projected.shape[-1]):
    pca = PCA(2) # project to 2 dimensions
    projected = pca.fit_transform(projected)
  # plot classes with labels
  plt.figure(figsize=(10, 10))
  plt.scatter(projected[:, 0], projected[:, 1], c=np.arange(len(classes)), cmap='tab20')
  for i, txt in enumerate(modelConfig['classes']):
    plt.annotate(txt, (projected[i, 0], projected[i, 1]))

  plt.tight_layout()
  plt.savefig('pca.png')
  plt.close()