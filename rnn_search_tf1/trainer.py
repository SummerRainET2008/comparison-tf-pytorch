import param
from rnn_search_tf1._model import Model
import tensorflow as tf
import optparse
import os
import pickle
import random
import time
from pa_nlp.tf_1x import nlp_tf
import numpy as np

class Trainer:
  def __init__(self):
    self._model = Model()
    self._input_x = tf.placeholder(tf.int32, [None, 50])
    self._input_y = tf.placeholder(tf.int32, [None, 50])
    self._loss = self._model(self._input_x, self._input_y)

    optimizer = tf.contrib.opt.LazyAdamOptimizer(param.lr)
    vars = tf.trainable_variables()
    gradients = optimizer.compute_gradients(
      self._loss, vars, colocate_gradients_with_ops=True
    )
    self._train_op = optimizer.apply_gradients(gradients)

    self._sess = nlp_tf.get_new_session()
    self._sess.run(tf.global_variables_initializer())

  def _train_one_batch(self, src, trg):
    [loss, _] = self._sess.run(
      fetches=[self._loss, self._train_op],
      feed_dict={
        self._input_x: src,
        self._input_y: trg
      }
    )

    return loss

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

        buff.append(
          [np.array(batch_src, np.int32), np.array(batch_tgt, np.int32)]
        )

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

