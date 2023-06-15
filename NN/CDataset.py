import tensorflow as tf
import tensorflow_datasets as tfds
from  NN import augmentations
from functools import lru_cache

class CDataset:
  def __init__(self, target_size=(224, 224)):
    self._splits, self._info = self._loadDataset()
    self._targetSize = target_size
    return

  def _loadDataset(self):
    (ds_train, ds_test), ds_info = tfds.load(
      'stanford_dogs',
      split=['train', 'test'],
      shuffle_files=True,
      as_supervised=False,
      with_info=True,
    )
    return {'train': ds_train, 'test': ds_test}, ds_info

  def _preprocessImage(self, data, augmF=None):
    processed_image = data['image']
    assert processed_image.dtype == tf.uint8, 'Expected image of type uint8'
    processed_image = augmF(processed_image)
    for image in processed_image:
      tf.assert_equal(tf.shape(image)[-1], 3)
      tf.assert_equal(tf.shape(image)[:-1], self._targetSize)
      tf.assert_equal(image.dtype, tf.float32)
      continue

    label = data['label']
    return processed_image, label
  
  def _preprocessFixBatch(self, images, label, N):
    images = tf.transpose(images, perm=[1, 0, 2, 3, 4])
    # images = [images[i] for i in range(N)] # this is not working, because tfds converts list to tensor
    return images, label

  def _prepare(self, split, augmF, N, batch_size, repeat=1, shuffle=0):
    assert split in self._splits.keys(), f'Unknown split: {split}'
    assert 0 < batch_size, f'Batch size must be positive, but got {batch_size}'

    dataset = self._splits[split]
    if shuffle: dataset = dataset.shuffle(shuffle * batch_size) # shuffle across N batches
    if 1 < repeat: dataset = dataset.repeat(repeat)
    
    dataset = dataset.map(
      lambda x: self._preprocessImage(x, augmF=augmF),
      num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    dataset = dataset.map(
      lambda x, y: self._preprocessFixBatch(x, y, N=N),
      num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset
  
  @property
  @lru_cache(maxsize=1)
  def classes_names(self):
    unformatted_labels = self._info.features['label'].names
    breed_names = [''.join(item.split('-')[1:]) for item in unformatted_labels]
    # replace '_' with ' '
    breed_names = [item.replace('_', ' ') for item in breed_names]
    # Make first letter capital
    breed_names = [item.capitalize() for item in breed_names]
    return breed_names
  
  # some predefined augmentations for dataset
  def train_dataset(self, augmF=None, N=1, batch_size=None, repeat=1, shuffle=5):
    if augmF is None:
      augmF = augmentations.augmentationsFor(target_size=self._targetSize, N=N)
    return self._prepare('train', augmF, N, batch_size, repeat=repeat, shuffle=shuffle)
  
  def test_dataset(self, batch_size=None):
    return self._prepare(
      split='test', batch_size=batch_size, N=1,
      augmF=augmentations.augmentationsFor(
        target_size=self._targetSize, N=1
      )
    )

if __name__ == "__main__":
  import cv2, json, os
  import numpy as np
  
  dataset = CDataset()
  print(dataset.classes_names)
  print(len(dataset.classes_names))
  
  ds_train = dataset.test_dataset(batch_size=2)
  for batch in ds_train.take(16):
    images, label = batch
    image = []
    for img in images:
      img = img.numpy()
      img = [img[i] for i in range(1)]
      img = np.concatenate(img, axis=0)
      image.append(img)
      continue

    label = label.numpy()[0]
    print(label, dataset.classes_names[label])

    image = np.concatenate(image, axis=1)
    cv2.imshow('image', image[..., ::-1])
    cv2.waitKey(0)
    continue