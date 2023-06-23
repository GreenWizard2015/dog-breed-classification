# Entry point for the gradio app (HuggingFace space)
from NN.model import model_from_config
from Utils.WandBUtils import CWBProject
import argparse
from UI import AppUI
import numpy as np

def modelFromRun(run):
  config = run.config
  model = model_from_config(config)
  model.load_weights(run.bestModel.pathTo())
  return dict(
    model=model,
    config=config,
    classes=config['classes'],
  )

def loadAllModels():
  WBProject = CWBProject('green_wizard/dog-breed-classification')
  runs = WBProject.groups(onlyBest=True) # dict of {model_name: CWBRun}
  # just in case, remove models with other than 224x224 image size
  runs = {k: v for k, v in runs.items() if v.config['imageSize'] == 224}
  return {k: modelFromRun(v) for k, v in runs.items()}

def predictor(models):
  def infer(model, inputImage):
    prediction = model['model'].predict([ inputImage[None] ])['probs'][0]
    classes = np.argsort(prediction)[::-1]
    # return class names and probabilities
    return [
      (model['classes'][i], prediction[i])
      for i in classes
    ]
  
  def predict(inputImage):
    assert inputImage.shape == (224, 224, 3), f'Expected image of shape (224, 224, 3), got {inputImage.shape}'
    assert inputImage.dtype == np.uint8, f'Expected image of dtype uint8, got {inputImage.dtype}'

    inputImage = inputImage.astype(np.float32) / 255.0
    results = {}
    for k, v in models.items():
      results[k] = infer(v, inputImage)
      continue
    return results
  return predict

def main(args):
  models = loadAllModels()
  print(f'Loaded {len(models)} models')
  
  app = AppUI(
    predictor=predictor(models),
  )
  app.queue() # enable queueing of requests/events
  app.launch(inline=False, server_port=args.port, server_name=args.host)
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=7860)
  parser.add_argument('--host', type=str, default=None)
  args = parser.parse_args()
  main(args)