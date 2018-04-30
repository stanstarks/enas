import os
import sys

import numpy as np
import tensorflow as tf

from src.sr.image_ops import conv
from src.sr.image_ops import fully_connected
from src.sr.image_ops import batch_norm
from src.sr.image_ops import relu
from src.sr.image_ops import max_pool
from src.sr.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops


class Model(object):
  def __init__(self,
               inputs,
               targets,
               cutout_size=None,
               batch_size=32,
               eval_batch_size=100,
               clip_mode=None,
               grad_bound=None,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.1,
               keep_prob=1.0,
               optim_algo=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               data_format="NHWC",
               name="generic_model",
               seed=None,
              ):
    """
    Args:
      lr_dec_every: number of epochs to decay
    """
    print "-" * 80
    print "Build model {}".format(name)

    self.cutout_size = cutout_size
    self.batch_size = batch_size
    self.eval_batch_size = eval_batch_size
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.l2_reg = l2_reg
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_rate = lr_dec_rate
    self.keep_prob = keep_prob
    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.data_format = data_format
    self.name = name
    self.seed = seed
    
    self.global_step = None
    self.valid_acc = None
    self.test_acc = None
    print "Build data ops"
    with tf.device("/cpu:0"):
      # training data
      self.num_train_examples = np.shape(inputs["train"])[0]
      self.num_train_batches = (
        self.num_train_examples + self.batch_size - 1) // self.batch_size
      x_train, y_train = tf.train.shuffle_batch(
        [inputs["train"], targets["train"]],
        batch_size=self.batch_size,
        capacity=50000,
        enqueue_many=True,
        min_after_dequeue=0,
        num_threads=16,
        seed=self.seed,
        allow_smaller_final_batch=True,
      )
      self.lr_dec_every = lr_dec_every * self.num_train_batches

      def _pre_process(x):
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = tf.random_crop(x, [32, 32, 3], seed=self.seed)
        x = tf.image.random_flip_left_right(x, seed=self.seed)
        if self.cutout_size is not None:
          mask = tf.ones([self.cutout_size, self.cutout_size], dtype=tf.int32)
          start = tf.random_uniform([2], minval=0, maxval=32, dtype=tf.int32)
          mask = tf.pad(mask, [[self.cutout_size + start[0], 32 - start[0]],
                               [self.cutout_size + start[1], 32 - start[1]]])
          mask = mask[self.cutout_size: self.cutout_size + 32,
                      self.cutout_size: self.cutout_size + 32]
          mask = tf.reshape(mask, [32, 32, 1])
          mask = tf.tile(mask, [1, 1, 3])
          x = tf.where(tf.equal(mask, 0), x=x, y=tf.zeros_like(x))
        if self.data_format == "NCHW":
          x = tf.transpose(x, [2, 0, 1])

        return x
      # self.x_train = tf.map_fn(_pre_process, x_train, back_prop=False)
      self.x_train = x_train
      self.y_train = y_train

      # valid data
      self.x_valid, self.y_valid = None, None
      if inputs["valid"] is not None:
        inputs["valid_original"] = np.copy(inputs["valid"])
        targets["valid_original"] = np.copy(targets["valid"])
        if self.data_format == "NCHW":
          inputs["valid"] = tf.transpose(inputs["valid"], [0, 3, 1, 2])
        self.num_valid_examples = np.shape(inputs["valid"])[0]
        self.num_valid_batches = (
          (self.num_valid_examples + self.eval_batch_size - 1)
          // self.eval_batch_size)
        self.x_valid, self.y_valid = tf.train.batch(
          [inputs["valid"], targets["valid"]],
          batch_size=self.eval_batch_size,
          capacity=5000,
          enqueue_many=True,
          num_threads=1,
          allow_smaller_final_batch=True,
        )

      # test data
      if self.data_format == "NCHW":
        inputs["test"] = tf.transpose(inputs["test"], [0, 3, 1, 2])
      self.num_test_examples = np.shape(inputs["test"])[0]
      self.num_test_batches = (
        (self.num_test_examples + self.eval_batch_size - 1)
        // self.eval_batch_size)
      self.x_test, self.y_test = tf.train.batch(
        [inputs["test"], targets["test"]],
        batch_size=self.eval_batch_size,
        capacity=10000,
        enqueue_many=True,
        num_threads=1,
        allow_smaller_final_batch=True,
      )

    # cache inputs and targets
    self.inputs = inputs
    self.targets = targets

  def eval_once(self, sess, eval_set, feed_dict=None, verbose=False):
    """Expects self.acc and self.global_step to be defined.

    Args:
      sess: tf.Session() or one of its wrap arounds.
      feed_dict: can be used to give more information to sess.run().
      eval_set: "valid" or "test"
    """

    assert self.global_step is not None
    global_step = sess.run(self.global_step)
    print "Eval at {}".format(global_step)
   
    if eval_set == "valid":
      assert self.x_valid is not None
      assert self.valid_psnr is not None
      num_examples = self.num_valid_examples
      num_batches = self.num_valid_batches
      acc_op = self.valid_psnr
    elif eval_set == "test":
      assert self.test_psnr is not None
      num_examples = self.num_test_examples
      num_batches = self.num_test_batches
      acc_op = self.test_psnr
    else:
      raise NotImplementedError("Unknown eval_set '{}'".format(eval_set))

    total_acc = 0
    total_exp = 0
    for batch_id in xrange(num_batches):
      acc = sess.run(acc_op, feed_dict=feed_dict)
      total_acc += acc
      total_exp += self.eval_batch_size
      if verbose:
        sys.stdout.write("\r{:<5d}/{:>5d}".format(total_acc, total_exp))
    if verbose:
      print ""
    print "{}_psnr: {:<6.4f}".format(
      eval_set, float(total_acc) / total_exp)
"""
  def _build_train(self):
    print "Build train graph"
    logits = self._model(self.x_train, True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, targets=self.y_train)
    self.loss = tf.reduce_mean(log_probs)

    self.train_preds = tf.argmax(logits, axis=1)
    self.train_preds = tf.to_int32(self.train_preds)
    self.train_acc = tf.equal(self.train_preds, self.y_train)
    self.train_acc = tf.to_int32(self.train_acc)

self.train_acc = tf.reduce_sum(self.train_acc)

    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print "-" * 80
    for var in tf_variables:
      print var

    self.global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")
    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.global_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas)

  def _build_valid(self):
    if self.x_valid is not None:
      print "-" * 80
      print "Build valid graph"
      logits = self._model(self.x_valid, False, reuse=True)
      self.valid_preds = tf.argmax(logits, axis=1)
      self.valid_preds = tf.to_int32(self.valid_preds)
      self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
      self.valid_acc = tf.to_int32(self.valid_acc)
      self.valid_acc = tf.reduce_sum(self.valid_acc)

  def _build_test(self):
    print "-" * 80
    print "Build test graph"
    logits = self._model(self.x_test, False, reuse=True)
    self.test_preds = tf.argmax(logits, axis=1)
    self.test_preds = tf.to_int32(self.test_preds)
    self.test_acc = tf.equal(self.test_preds, self.y_test)
    self.test_acc = tf.to_int32(self.test_acc)
    self.test_acc = tf.reduce_sum(self.test_acc)

  def build_valid_rl(self, shuffle=False):
    print "-" * 80
    print "Build valid graph on shuffled data"
    with tf.device("/cpu:0"):
      # shuffled valid data: for choosing validation model
      if not shuffle and self.data_format == "NCHW":
        self.inputs["valid_original"] = np.transpose(
          self.inputs["valid_original"], [0, 3, 1, 2])
      x_valid_shuffle, y_valid_shuffle = tf.train.shuffle_batch(
        [self.inputs["valid_original"], self.targets["valid_original"]],
        batch_size=self.batch_size,
        capacity=25000,
        enqueue_many=True,
        min_after_dequeue=0,
        num_threads=16,
        seed=self.seed,
        allow_smaller_final_batch=True,
      )

      def _pre_process(x):
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = tf.random_crop(x, [32, 32, 3], seed=self.seed)
        x = tf.image.random_flip_left_right(x, seed=self.seed)
        if self.data_format == "NCHW":
          x = tf.transpose(x, [2, 0, 1])

        return x

      if shuffle:
        x_valid_shuffle = tf.map_fn(_pre_process, x_valid_shuffle,
                                    back_prop=False)

    logits = self._model(x_valid_shuffle, False, reuse=True)
    valid_shuffle_preds = tf.argmax(logits, axis=1)
    valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
    self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
    self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)
"""
  def _model(self, inputs, is_training, reuse=None):
    raise NotImplementedError("Abstract method")
