import tensorflow as tf
import numpy as np
import albumentations as A

def withAugmentations(transforms, finalTransform):
  N = len(transforms)
  mainTransform, *otherTransforms = transforms
  def aug_fn(image):
    image = mainImg = mainTransform(image=image)['image']
    res = [mainImg]
    for transform in otherTransforms:
      image = transform(image=image)['image']
      res.append(image)
      continue

    res = [finalTransform(image=img)['image'] for img in res]
    return res
  
  def F(image):
    tf.assert_rank(image, 3)
    return tf.numpy_function(func=aug_fn, inp=[image], Tout=[tf.float32] * N)
  
  return F

def augmentationsFor(target_size=224, N=2):
  if isinstance(target_size, int): target_size = (target_size, target_size)
  assert isinstance(target_size, tuple), 'target_size must be tuple or int'
  assert len(target_size) == 2, 'target_size must be tuple of length 2'

  mainAugmentations = A.Compose([ # light augmentations for main image
    A.RandomResizedCrop(target_size[0], target_size[1], scale=(0.5, 2.0), always_apply=True),
    A.Rotate(limit=45),
  ])

  AOther = A.Compose([ # heavy augmentations for other images
    A.MultiplicativeNoise(multiplier=(0.8, 1.2)),
    A.RandomBrightnessContrast(brightness_limit=0.025, contrast_limit=0.025),
    A.Downscale(scale_min=0.9, scale_max=0.9),
    A.Sharpen(),
    A.Rotate(limit=25),
  ])
  return withAugmentations(
    transforms=[mainAugmentations] + [AOther] * (N - 1),
    finalTransform=A.Compose([ A.ToFloat() ])
  )

def noAugmentation(target_size=224):
  if isinstance(target_size, int): target_size = (target_size, target_size)
  assert isinstance(target_size, tuple), 'target_size must be tuple or int'
  assert len(target_size) == 2, 'target_size must be tuple of length 2'

  mainAugmentations = A.Compose([
    A.Resize(target_size[0], target_size[1], always_apply=True),
  ])
  return withAugmentations(
    transforms=[mainAugmentations],
    finalTransform=A.Compose([ A.ToFloat() ])
  )