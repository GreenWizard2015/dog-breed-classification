import tensorflow as tf

class CHyperSoftmaxLayer(tf.keras.layers.Layer):
  def __init__(self, classes, **kwargs):
    super().__init__(**kwargs)
    self._classes = classes
    return
  
  def build(self, input_shape):
    C = input_shape[-1]
    self._W = tf.Variable(
      initial_value=tf.random_normal_initializer()(shape=(self._classes, C)),
      trainable=True,
      name='W'
    )
    self._P = tf.Variable(initial_value=1.0, trainable=True, name='P')
    return super().build(input_shape)

  def call(self, x):
    emb = tf.linalg.l2_normalize(self._W, axis=-1)
    normalizedX, L = tf.linalg.normalize(x, axis=-1)

    P = tf.clip_by_value(self._P, 1e-1, 1e+1)
    cdist = tf.matmul(normalizedX, emb, transpose_b=True)
    logits = tf.pow(cdist + 1.0, P) * L

    return tf.nn.softmax(logits, axis=-1)
