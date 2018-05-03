import os
import sys
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from random import randint

def _bytes_feature(img):
  value = tf.compat.as_bytes(img.tostring())
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _get_image(img, coord, size, pad):
  x, y = coord
  return img[x*size:(x+1)*size+2*pad, y*size:(y+1)*size+2*pad]

def _read_data(data_path, img_size, ratio, track='bicubic', pad=4):
  """Reads DIV2K data. Always returns NHWC format.

  Code adopted from EDSR-Tensorflow (https://github.com/jmiller656/EDSR-Tensorflow)

  Returns:
    inputs: np tensor of size [N, H, W, C]
    targets: np tensor of size [N, H, W, C]
  """

  # pre-load the whole set
  inputs, targets = [], []
  dataset_name = 'DIV2K_train'
  hr_path = os.path.join(data_path,
                         '_'.join((dataset_name, 'HR')))
  lr_path = os.path.join(data_path,
                         '_'.join((dataset_name, 'LR', track)),
                         'X%d' % ratio)
  lr_end = 'x%d.png' % ratio
  hr_files = os.listdir(hr_path)
  for hr_name in hr_files:
    hr_fullname = os.path.join(hr_path, hr_name)
    lr_name = os.path.splitext(hr_name)[0] + lr_end
    lr_fullname = os.path.join(lr_path, lr_name)
    lr_img = imread(lr_fullname)
    hr_img = imread(hr_fullname)
    x, y, _ = lr_img.shape
    x -= pad * 2
    y -= pad * 2

    # scale inputs
    for i in range(x / img_size):
      for j in range(y / img_size):
        inputs.append(_get_image(lr_img, (i, j), img_size, pad))
        targets.append(_get_image(hr_img, (i, j), img_size*ratio, pad*ratio))
  inputs = np.concatenate(inputs, axis=0)
  targets = np.concatenate(targets, axis=0)
  inputs = np.reshape(inputs, [-1, img_size+2*pad, img_size+2*pad, 3])
  targets = np.reshape(targets, [-1, (img_size+2*pad)*ratio,
                                 (img_size+2*pad)*ratio, 3])

  # normalize
  inputs = inputs.astype(np.float32)
  inputs = (inputs - 127.0) / 127.0

  return inputs, targets

def _augment_helper(inp, target, img_size, pad, ratio):
  offset_x = randint(0, pad*2-1)
  offset_y = randint(0, pad*2-1)
  inp = tf.slice(inp, [offset_x, offset_y, 0], [img_size, img_size, 3])
  target = tf.slice(target, [offset_x*ratio, offset_y*ratio, 0],
                    [img_size*ratio, img_size*ratio, 3])
  target = tf.cast(target, dtype=tf.float32)
  return inp, target

def parse_fn(example, img_size=32, pad=4, ratio=4):
  example_fmt = {
    'input': tf.FixedLenFeature((), tf.string, ""),
    'target': tf.FixedLenFeature((), tf.string, "")
  }
  parsed = tf.parse_single_example(example, example_fmt)
  inp = tf.decode_raw(parsed['input'], tf.float32)
  target = tf.decode_raw(parsed['target'], tf.uint8)
  inp = tf.reshape(inp, [img_size+pad*2, img_size+pad*2, 3])
  target = tf.reshape(target, [(img_size+pad*2)*ratio, (img_size+pad*2)*ratio, 3])
  inp, target = _augment_helper(inp, target, img_size, pad, ratio)
  return inp, target

def input_fn(data_path, filename):
  filename = os.path.join(data_path, filename + '.tfrecord')
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.shuffle(buffer_size=3000)
  dataset = dataset.repeat(3000)
  dataset = dataset.map(map_func=parse_fn,
                        num_parallel_calls=16)
  return dataset

def read_data(data_path, num_valids=20000, num_tests=8000, img_size=32, ratio=4):
  print "-" * 80
  print "Reading data"

  datafile = os.path.join(data_path, 'train.tfrecord')
  if not os.path.exists(datafile):
    print "creating tfrecord"
    inputs, targets = _read_data(data_path, img_size, ratio)

    if num_valids:
      inputs_valid = inputs[-num_valids:]
      targets_valid = targets[-num_valids:]
      inputs = inputs[:-num_valids]
      targets = targets[:-num_valids]
      write_tfrecord(data_path, inputs_valid, targets_valid, 'valid')

    if num_tests:
      inputs_test = inputs[-num_tests:]
      targets_test = targets[-num_tests:]
      inputs = inputs[:-num_tests]
      targets = targets[:-num_tests]
      write_tfrecord(data_path, inputs_test, targets_test, 'test')

    write_tfrecord(data_path, inputs, targets, 'train')

  datasets = {}
  datasets['train'] = input_fn(data_path, 'train')
  if num_valids:
    datasets['valid'] = input_fn(data_path, 'valid')
  else:
    datasets['valid'] = None
  datasets['test'] = input_fn(data_path, 'test')
  return datasets


def write_tfrecord(data_path, x, y, dataset):
  filename = os.path.join(data_path, dataset+'.tfrecord')
  writer = tf.python_io.TFRecordWriter(filename)

  print dataset + ' num of examples: %d' % x.shape[0]
  for i in range(x.shape[0]):
    feature = {
      'input': _bytes_feature(x[i]),
      'target': _bytes_feature(y[i]),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

  writer.close()

