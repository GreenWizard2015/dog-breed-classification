from NN.CDataset import CDataset
from NN.model import model_from_config
from NN.CTrainer import CTrainer

import tensorflow as tf
import argparse, json, os

def _toRunName(args):
  nameParts = []
  if args.model_class_size: 
    nameParts.append(f'HS-{args.model_class_size} ({args.model_hs_type})')
  else:
    nameParts.append('softmax')

  nameParts.append(f'{args.views} views')
  nameParts.append(f'{args.model_input}x{args.model_input}')
  nameParts.append('grayscale' if args.model_grayscale else 'RGB')
  return ', '.join(nameParts)

def main(args):
  NViews = args.views
  dataset = CDataset( target_size=(args.model_input, args.model_input) )

  modelConfig = {
    'classes': dataset.classes_names,
    'classesN': len(dataset.classes_names),
    'imageSize': args.model_input,
    'classVecSize': args.model_class_size if args.model_class_size else None,
    'grayscale': bool(args.model_grayscale),
    'HS type': args.model_hs_type,
  }
  print(json.dumps(modelConfig, indent=2)) # for debug

  model = model_from_config(modelConfig)
  model.summary() # for debug
  
  trainer = CTrainer(model=model, NViews=NViews)
  trainer.compile(loss=None, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

  folder = args.folder
  latestModel = os.path.join(folder, 'model-latest.h5')
  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
      filepath=latestModel,
      save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1
    ),
    tf.keras.callbacks.TerminateOnNaN(),
  ]
  
  USE_WANDB = args.wandb_user and args.wandb_project
  if USE_WANDB:
    import wandb

    if args.wandb_run is None: args.wandb_run = _toRunName(args)
    wandb.init(
      entity=args.wandb_user,
      project=args.wandb_project,
      name=args.wandb_run if args.wandb_run else None,
      config=modelConfig
    )
    # track model metrics only
    callbacks.append(wandb.keras.WandbCallback(
      save_model=False, # save model to wandb manually
      save_graph=False,
    ))
    # models are saved manually, because wandb callback can't handle complex model
    pass

  try:
    trainer.fit(
      dataset.train_dataset(batch_size=args.batch_size, repeat=args.repeat, N=NViews),
      validation_data=dataset.test_dataset(batch_size=128),
      verbose=2,
      epochs=args.epochs,
      callbacks=callbacks,
    )
  finally:
    if USE_WANDB:
      files = [latestModel]
      for f in files:
        wandb.log_artifact(f, type='bytes')
      wandb.finish()
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--folder', type=str, help='Folder to save model and logs', default=os.getcwd())
  parser.add_argument('--wandb-user', type=str, help='Wandb user name (optional)')
  parser.add_argument('--wandb-project', type=str, help='Wandb entity name (optional)')
  parser.add_argument('--wandb-run', type=str, help='Wandb run name (optional)')

  parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
  parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
  parser.add_argument('--repeat', type=int, default=1, help='How many times to repeat the train dataset')
  parser.add_argument('--views', type=int, default=3, help='How many views to use for each image')

  parser.add_argument('--model-input', type=int, default=224, help='Model input size')
  parser.add_argument('--model-class-size', type=int, default=32, help='Model class vector size. Will be used softmax if 0.')
  # hyperspherical softmax type
  parser.add_argument(
    '--model-hs-type', type=str, default='base',
    help='Hyperspherical softmax type ("base", "learnable", "learnable invertible")'
  )
  # grayscale flag
  parser.add_argument('--model-grayscale', action='store_true', help='Use grayscale images during inference')

  main(parser.parse_args())