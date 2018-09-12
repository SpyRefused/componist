from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

tf.__version__


model = keras.models.load_model('my_model.h5')
