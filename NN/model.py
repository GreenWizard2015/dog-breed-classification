import tensorflow as tf
import tensorflow.keras.layers as L
from .CHyperSoftmaxLayer import CHyperSoftmaxLayer

def _MobileNetV2(data):
  dataIn = L.Input(shape=data.shape[1:])
  x = (2.0 * dataIn) - 1.0 # normalize to [-1, 1]
  base_model = tf.keras.applications.MobileNetV2(input_shape=data.shape[1:], include_top=False)
  base_model.trainable = False
  x = base_model(x, training=False)
  # reduce the output size
  x = L.AvgPool2D(pool_size=(7, 7))(x)

  model = tf.keras.Model(inputs=dataIn, outputs=x)
  return model(data) # return the output of the backbone

def _ResNet152(data):
  dataIn = L.Input(shape=data.shape[1:])
  base_model = tf.keras.applications.ResNet152(input_shape=data.shape[1:], include_top=False)
  base_model.trainable = False
  # dataIn in [0, 1], but resnet expects [0.0, 255.0]
  x = tf.keras.applications.resnet.preprocess_input(dataIn * 255.0)
  x = base_model(x, training=False)

  assert x.shape[1:] == (7, 7, 2048), 'Expected shape (7, 7, 2048), got {}'.format(x.shape[1:])
  # reduce the output size
  x = L.AvgPool2D(pool_size=(7, 7))(x)
  model = tf.keras.Model(inputs=dataIn, outputs=x)
  return model(data) # return the output of the backbone

def _ResNet50(data):
  dataIn = L.Input(shape=data.shape[1:])
  base_model = tf.keras.applications.ResNet50(input_shape=data.shape[1:], include_top=False)
  base_model.trainable = False
  # dataIn in [0, 1], but resnet expects [0.0, 255.0]
  x = tf.keras.applications.resnet.preprocess_input(dataIn * 255.0)
  x = base_model(x, training=False)

  assert x.shape[1:] == (7, 7, 2048), 'Expected shape (7, 7, 2048), got {}'.format(x.shape[1:])
  # reduce the output size
  x = L.AvgPool2D(pool_size=(7, 7))(x)
  model = tf.keras.Model(inputs=dataIn, outputs=x)
  return model(data) # return the output of the backbone

def _EfficientNetB5(data):
  dataIn = L.Input(shape=data.shape[1:])
  base_model = tf.keras.applications.EfficientNetB5(input_shape=data.shape[1:], include_top=False)
  base_model.trainable = False
  # dataIn in [0, 1], but resnet expects [0.0, 255.0]
  x = tf.keras.applications.efficientnet.preprocess_input(dataIn * 255.0)
  x = base_model(x, training=False)

  assert x.shape[1:] == (7, 7, 2048), 'Expected shape (7, 7, 2048), got {}'.format(x.shape[1:])
  # reduce the output size
  x = L.AvgPool2D(pool_size=(7, 7))(x)
  model = tf.keras.Model(inputs=dataIn, outputs=x)
  return model(data) # return the output of the backbone

BACKBONES = {
  'MobileNetV2': _MobileNetV2,
  'ResNet152': _ResNet152,
  'ResNet50': _ResNet50,
  'EfficientNetB5': _EfficientNetB5,
}

# convert to grayscale but keep 3 channels
def _rgb_to_grayscale(x):
  x = tf.image.rgb_to_grayscale(x)
  x = tf.repeat(x, 3, axis=-1)
  return x

def _model_body(res):
  # MLP classification head
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

def _createModel(imageSize, headF, grayscale, backboneF):
  res = images = L.Input(shape=(imageSize, imageSize, 3))
  if grayscale: res = L.Lambda(_rgb_to_grayscale)(res)

  classVec, probs = headF(
    _model_body(
      backboneF(res)
    )
  )
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

  # find the backbone in the BACKBONES dict, case insensitive
  backboneName = config.get('backbone', 'MobileNetV2').lower()
  matched = [
    backboneF
    for name, backboneF in BACKBONES.items()
    if name.lower() == backboneName
  ]
  assert len(matched) == 1, f'Unknown backbone: {backboneName}'
  backboneF = matched[0]

  return _createModel(
    imageSize=config['imageSize'],
    grayscale=config['grayscale'],
    headF=headF,
    backboneF=backboneF,
  )