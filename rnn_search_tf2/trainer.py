import param
from rnn_search_tf2._model import Model
import tensorflow as tf
import optparse
import os
import pickle
import random
import time
from pa_nlp.tf_2x.lamb_optimizer import LAMBOptimizer

class Trainer:
  def __init__(self):
    self._model = Model()
    # self._optimizer = tf.keras.optimizers.Adam(learning_rate=param.lr)
    self._optimizer = LAMBOptimizer(learning_rate=param.lr)
    # self._optimizer = tf.keras.optimizers.RMSprop(learning_rate=param.lr)

  @tf.function(
    input_signature=(
      tf.TensorSpec(shape=[None, None], dtype=tf.int32),
      tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    )
  )
  def _train_one_batch(self, src, trg):
    print(f"retracing _train_one_batch({src.shape}, {trg.shape})")
    with tf.GradientTape() as tape:
      loss = self._model(src, trg)
    batch_loss = (loss / tf.cast(tf.shape(trg)[1], tf.float32))
    self._apply_optimizer(tape, loss)
    return batch_loss

  def _apply_optimizer(self, tape, loss, norm=5):
    print(f"retracing _apply_opt({tape}, {loss.shape})")
    # print(f"apply_optimizer")
    variables = self._model.trainable_variables
    gradients = tape.gradient(loss, variables)
    cropped_g, _ = tf.clip_by_global_norm(gradients, norm)

    self._optimizer.apply_gradients(zip(cropped_g, variables))

    return loss

  def train(self):
    for batch_id, [b_src, b_tgt] in enumerate(self.get_batch_data()):
      start_time = time.time()
      batch_loss = self._train_one_batch(b_src, b_tgt)
      duration = time.time() - start_time
      print(f"batch[{batch_id}]: loss: {batch_loss}, time: {duration} sec.")

  def get_batch_data(self):
    data_src, data_tgt = pickle.load(open("rnn_search_tf2/input.data", "rb"))
    data = list(zip(data_src, data_tgt))
    buff = []

    for _ in range(param.epoch_num):
      random.shuffle(data)
      for p in range(0, len(data), param.batch_size):
        batch_data = data[p: p + param.batch_size]
        batch_src = [e[0] for e in batch_data]
        batch_tgt = [e[1] for e in batch_data]

        buff.append([tf.convert_to_tensor(batch_src, tf.int32),
                     tf.convert_to_tensor(batch_tgt, tf.int32)])

    print(f"buff.size: {len(buff)}")
    yield from buff

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")
  (options, args) = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  trainer = Trainer()
  trainer.train()

if __name__ == '__main__':
  main()

