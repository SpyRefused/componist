# 1) Save and restore models
# Model progress can be saved during—and after—training. This means a model can resume where it left
# off and avoid long training times. Saving also means you can share your model and others can recreate your work.
# When publishing research models and techniques, most machine learning practitioners share:
#  - code to create the model, and
#  - the trained weights, or parameters, for the model Sharing this data helps
# others understand how the model works and try it themselves with new data.

# 1.1) Options
# There are different ways to save TensorFlow models—depending on the API you're using.
# This guide uses tf.keras, a high-level API to build and train models in TensorFlow. For other approaches,
# see the TensorFlow Save and Restore guide or Saving in eager.


# 2) Setup
# 2.1) Installs and imports
# Install and import TensorFlow and dependencies:

# !pip install -q h5py pyyaml

# 2.1) Get an example dataset
# We'll use the MNIST dataset to train our model to demonstrate saving weights.
# To speed up these demonstration runs, only use the first 1000 examples:

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


