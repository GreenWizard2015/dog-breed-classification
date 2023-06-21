import tensorflow as tf
import tensorflow.keras.layers as L

class CBaseHyperSoftmax(tf.keras.layers.Layer):
  def __init__(self, classes, dims, mapping, **kwargs):
    super().__init__(**kwargs)
    self._mapping = mapping
    
    self._W = tf.Variable(
      initial_value=tf.random_normal_initializer()(shape=(classes, dims)),
      trainable=True,
      name='W'
    )
    return

  @property
  def embeddings(self): return tf.linalg.l2_normalize(self._W, axis=-1)

  def _logits(self, x):
    normalizedX = tf.linalg.l2_normalize(x, axis=-1)
    similarity = tf.matmul(normalizedX, self.embeddings, transpose_b=True)

    logits = self._mapping(similarity)
    tf.assert_equal(tf.shape(logits), tf.shape(similarity))
    return logits
    
  def call(self, x):
    return tf.nn.softmax(self._logits(x), axis=-1)
  
  def classesSimilarity(self): return self._logits(self._W)
# End of CBaseHyperSoftmax

class CHyperSoftmaxLayer(tf.keras.layers.Layer):
  def __init__(self, classes, **kwargs):
    super().__init__(**kwargs)
    self._classes = classes

    # Network for mapping similarity to logits
    # This mapping is constrained to be strictly increasing, so f(x) <= f(x + dx) <= f(x + dx * 2) ...
    NN = tf.keras.constraints.NonNeg()
    self._map = tf.keras.Sequential([
      L.InputLayer(input_shape=(1,)),
      # relu6 is used just to force finding a non-trivial solution
      L.Dense(16, 'relu6', kernel_constraint=NN),
      L.Dense(16, 'relu6', kernel_constraint=NN),
      L.Dense(1, 'relu', kernel_constraint=NN),
    ], name='mapping')
    return

  def _mapping(self, similarity):
    x = self._map(tf.reshape(similarity, (-1, 1)))
    return tf.reshape(x, tf.shape(similarity))
    
  def build(self, input_shape):
    C = input_shape[-1]
    self._hsl = CBaseHyperSoftmax(self._classes, C, self._mapping)
    return super().build(input_shape)

  def call(self, x): return self._hsl(x)

  # helper functions
  def classesSimilarity(self): return self._hsl.classesSimilarity()
  def mapping(self, values): return self._mapping(values)

  @property
  def embeddings(self): return self._hsl.embeddings
# End of CHyperSoftmaxLayer