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


# 2.2) Define a model
# Let's build a simple model we'll use to demonstrate saving and loading weights.

# Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


# Create a basic model instance
model = create_model()
model.summary()

# 3) Save checkpoints during training
# The primary use case is to automatically save checkpoints during and at the end of training.
# This way you can use a trained model without having to retrain it, or pick-up training where
# you left of—in case the training process was interrupted.
# tf.keras.callbacks.ModelCheckpoint is a callback that performs this task.
# The callback takes a couple of arguments to configure checkpointing.

# 3.1) Checkpoint callback usage
# Train the model and pass it the ModelCheckpoint callback:

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # pass callback to training

# This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:
# !ls {checkpoint_dir}

# Create a new, untrained model. When restoring a model from only weights, you must have a model with the same
# architecture as the original model. Since it's the same model architecture,
# we can share weights despite that it's a different instance of the model.
# Now rebuild a fresh, untrained model, and evaluate it on the test set.
# An untrained model will perform at chance levels (~10% accuracy):

model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# Then load the weights from the checkpoint, and re-evaluate:

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# 3.2) Checkpoint callback options
# The callback provides several options to give the resulting checkpoints unique names,
# and adjust the checkpointing frequency.
#
# Train a new model, and save uniquely named checkpoints once every 5-epochs:

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs=50, callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

# Now, have a look at the resulting checkpoints (sorting by modification date):

import pathlib

# Sort the checkpoints by modification time.
checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
checkpoints = sorted(checkpoints, key=lambda cp: cp.stat().st_mtime)
checkpoints = [cp.with_suffix('') for cp in checkpoints]
latest = str(checkpoints[-1])
checkpoints
