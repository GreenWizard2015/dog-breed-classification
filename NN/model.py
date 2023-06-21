import tensorflow as tf
import tensorflow.keras.layers as L
from .CHyperSoftmaxLayer import CHyperSoftmaxLayer

def _featuresProvider(shape):
  base_model = tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False)
  base_model.trainable = False
  return base_model

# convert to grayscale but keep 3 channels
def _rgb_to_grayscale(x):
  x = tf.image.rgb_to_grayscale(x)
  x = tf.repeat(x, 3, axis=-1)
  return x

def _model_body(res):
  res = L.AvgPool2D(pool_size=(7, 7))(res)
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
  return res

def _createModel(imageSize, headF, grayscale=False):
  res = images = L.Input(shape=(imageSize, imageSize, 3))
  if grayscale: res = L.Lambda(_rgb_to_grayscale)(res)

  fp = _featuresProvider(res.shape[1:])
  res = _model_body(
    fp([(2.0 * res) - 1.0], training=False)
  )

  classVec, probs = headF(res)
  return tf.keras.Model(
    inputs=[images],
    outputs={
      'classVec': classVec,
      'probs': probs,
    }
  )

def _simpleSoftmax(classesN):
  def headF(res):
    classVec = L.Dense(classesN, activation='relu')(res)
    probs = L.Softmax()(classVec)
    return classVec, probs
  return headF

def _HS_base(classesN, classVecSize):
  def headF(res):
    classVec = L.Dense(classVecSize)(res)
    probs = CHyperSoftmaxLayer(classes=classesN)(classVec)
    return classVec, probs
  return headF

def model_from_config(config):
  classesN = config['classesN']
  classVecSize = config['classVecSize']
  headF = _simpleSoftmax(classesN)

  if 0 < classVecSize: # use hyperspherical softmax
    HSTypeMapping = {
      'base': _HS_base,
    }
    HSType = HSTypeMapping.get(config['HS type'].lower(), None)
    assert HSType is not None, f'Unknown HS type: {config["HS type"]}'
    headF = HSType(classesN=classesN, classVecSize=classVecSize)
    pass

  return _createModel(
    imageSize=config['imageSize'],
    grayscale=config['grayscale'],
    headF=headF,
  )