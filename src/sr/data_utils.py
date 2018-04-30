import os
import sys
import numpy as np
import tensorflow as tf
from scipy.misc import imread


def _get_image(img, coord, size):
  x, y = coord
  return img[x*size:(x+1)*size, y*size:(y+1)*size]


def _read_data(data_path, img_size, ratio, track='bicubic'):
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
                         '_'.join(dataset_name, 'HR'))
  lr_path = os.path.join(data_path,
                         '_'.join(dataset_name, 'LR', track),
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
    # offset = randint(0, img_size)
    for i in range(x / img_size):
      for j in range(y / img_size):
        inputs.append(_get_image(lr_img, (i, j), img_size))
        targets.append(_get_image(hr_img, (i, j), img_size * ratio))

  inputs = np.concatenate(inputs, axis=0)
  targets = np.concatenate(targets, axis=0)
  inputs = np.reshape(inputs, [-1, 3, img_size, img_size])
  targets = np.reshape(inputs, [-1, 3, img_size * ratio, img_size * ratio])

  return inputs, targets


def read_data(data_path, num_valids=8000, num_tests=8000, img_size=32, ratio=4):
  print "-" * 80
  print "Reading data"

  inputs, targets = {}, {}

  inputs["train"], targets["train"] = _read_data(data_path, img_size, ratio)

  if num_valids:
    inputs["valid"] = inputs["train"][-num_valids:]
    targets["valid"] = targets["train"][-num_valids:]

    inputs["train"] = inputs["train"][:-num_valids]
    targets["train"] = targets["train"][:-num_valids]
  else:
    inputs["valid"], targets["valid"] = None, None

  if num_tests:
    inputs["test"] = inputs["train"][-num_tests:]
    targets["test"] = targets["train"][-num_tests:]

    inputs["train"] = inputs["train"][:-num_tests]
    targets["train"] = targets["train"][:-num_tests]
  else:
    inputs["test"], targets["test"] = None, None

  print "Prepropcess: [subtract mean: 127], [divide std: 127]"

  mean = 127
  std = 127
  inputs["train"] = (inputs["train"] - mean) / std
  if num_valids:
    inputs["valid"] = (inputs["valid"] - mean) / std
  inputs["test"] = (inputs["test"] - mean) / std

  return inputs, targets
