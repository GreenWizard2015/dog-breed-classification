import tensorflow as tf
import tensorflow.keras.layers as L
from .CHyperSoftmaxLayer import CHyperSoftmaxLayer

def _featuresProvider(shape):
  base_model = tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False)
  base_model.trainable = False
  return base_model

# if classVecSize is None, then will be used simple softmax
def createModel(imageSize, classesN, classVecSize, grayscale=False):
  res = images = L.Input(shape=(imageSize, imageSize, 3))

  if grayscale:
    # convert to grayscale but keep 3 channels
    def _rgb_to_grayscale(x):
      x = tf.image.rgb_to_grayscale(x)
      x = tf.repeat(x, 3, axis=-1)
      return x
    res = L.Lambda(_rgb_to_grayscale)(res)
    pass

  fp = _featuresProvider(res.shape[1:])
  # 0..1 -> -1..1
  features = fp([(2.0 * res) - 1.0], training=False)

  res = L.AvgPool2D(pool_size=(7, 7))(features)
  res = L.Flatten()(res)
  for sz in [256, 128, 64]:
    res = L.Dense(sz, activation='relu')(res)

    skip = L.BatchNormalization()(res)
    for _ in range(3):
      skip = L.Dropout(0.05)(skip)
      skip = L.Dense(sz, activation='relu')(skip)
      continue
    skip = L.BatchNormalization()(skip)

    res = L.Add()([res, skip])
    continue

  if classVecSize is None:
    classVec = L.Dense(classesN, activation='relu')(res)
    probs = L.Softmax()(classVec)
  else:
    classVec = L.Dense(classVecSize)(res)
    probs = CHyperSoftmaxLayer(classes=classesN)(classVec)
    pass

  return tf.keras.Model(
    inputs=[images],
    outputs={
      'classVec': classVec,
      'probs': probs,
    }
  )

def model_from_config(config):
  return createModel(
    imageSize=config['imageSize'],
    classesN=config['classesN'],
    classVecSize=config['classVecSize'],
    grayscale=config['grayscale'],
  )