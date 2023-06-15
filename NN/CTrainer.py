import tensorflow as tf
from utils import FakeObject

class CTrainer(tf.keras.Model):
  def __init__(self, model, NViews, **kwargs):
    super().__init__(**kwargs)
    self._NViews = NViews

    self._model = model
    self._loss = tf.keras.metrics.Mean(name="loss")
    self._CELoss = tf.keras.metrics.Mean(name="CE")
    self._KLLoss = tf.keras.metrics.Mean(name="KL")

    self._accuracy = tf.keras.metrics.Accuracy(name="accuracy")
    self._topK = tf.keras.metrics.Mean(name="top-K")
    return

  @tf.function
  def call(self, X, training=False):
    return self._model(tf.concat(X, axis=0), training=training)

  def _infer(self, imagesList, training):
    pred = self._model([tf.concat(imagesList, axis=0)], training=training)
    splits = [tf.shape(imagesList[0])[0]] * len(imagesList)

    return FakeObject({
      name: tf.split(x, num_or_size_splits=splits, axis=0)
      for name, x in pred.items()
    })

  def _calcCELoss(self, data, predictions):
    CE = []
    for pred in predictions:
      CE.append(tf.losses.sparse_categorical_crossentropy(data.labels, pred))
      continue
    return tf.concat(CE, axis=0)

  def _calcKLLoss(self, predictions):
    KL = []
    anchor = predictions[0]
    for pred in predictions[1:]:
      KL.append(tf.losses.kl_divergence(anchor, pred))
      continue
    return tf.concat(KL, axis=0)

  def _updateAccuracy(self, data, predictions):
    predTop1 = [tf.argmax(pred, axis=-1, output_type=data.labels.dtype) for pred in predictions]
    self._accuracy.update_state(
      y_true=tf.concat([data.labels] * len(data.images), axis=0),
      y_pred=tf.concat(predTop1, axis=0)
    )
    return

  def _updateTopK(self, data, predictions):
    topK = []
    labelsOHE = tf.one_hot(data.labels, tf.shape(predictions[0])[-1])
    for pred in predictions:
      sortedInd = tf.argsort(pred, direction='DESCENDING')
      sortedInd = tf.cast(tf.argsort(sortedInd), tf.float32)
      tf.assert_equal(tf.shape(labelsOHE), tf.shape(sortedInd))
      topK.append(tf.reduce_max(sortedInd * labelsOHE, axis=-1))
      continue
    self._topK.update_state(tf.concat(topK, axis=0))
    return

  def _calcLoss(self, data):
    predictions = self._infer(data.images, training=True)
    addedLoss = sum(self._model.losses, 0.0)
    CELoss = self._calcCELoss(data, predictions.probs)
    KLLoss = self._calcKLLoss(predictions.probs)
    losses = [addedLoss, CELoss, KLLoss]
    ############
    self._updateAccuracy(data, predictions.probs)
    self._updateTopK(data, predictions.probs)
    ############
    totalLoss = sum([tf.reduce_mean(x) for x in losses])
    self._loss.update_state(totalLoss)
    self._CELoss.update_state(CELoss)
    self._KLLoss.update_state(KLLoss)
    return totalLoss

  @tf.function
  def train_step(self, data):
    images, (labels) = data
    images = [images[i] for i in range(self._NViews)]
    batch = FakeObject({ 'images': images, 'labels': labels, })
    with tf.GradientTape() as tape:
      totalLoss = self._calcLoss(batch)

    self.optimizer.minimize(totalLoss, self._model.trainable_variables, tape=tape)
    return {x.name: x.result() for x in [
      self._loss, self._CELoss, self._KLLoss,
      self._accuracy, self._topK
    ]}

  @tf.function
  def test_step(self, data):
    images, labels = data
    images = [images[i] for i in range(1)]
    data = FakeObject({ 'images': images, 'labels': labels, })

    predictions = self._infer(data.images, training=False)
    CELoss = self._calcCELoss(data, predictions.probs)
    losses = [CELoss]
    ############
    self._updateAccuracy(data, predictions.probs)
    self._updateTopK(data, predictions.probs)
    ############
    totalLoss = sum([tf.reduce_mean(x) for x in losses])
    self._loss.update_state(totalLoss)
    self._CELoss.update_state(CELoss)

    return {x.name: x.result() for x in [
      self._loss,
      self._accuracy, self._topK
    ]}