# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

# Loading the dataset returns four NumPy arrays:
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# The train_images and train_labels arrays are the training set—the data the model uses to learn.
# The model is tested against the test set, the test_images, and test_labels arrays

# The images are 28x28 NumPy arrays, with pixel values ranging between 0 and 255.
# The labels are an array of integers, ranging from 0 to 9.
# These correspond to the class of clothing the image represents:

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# The data must be preprocessed before training the network.
# If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)

# We scale these values to a range of 0 to 1 before feeding to the neural network model.
# For this, cast the datatype of the image components from an integer to a float, and divide by 255.
# Here's the function to preprocess the images:
# It's important that the training set and the testing set are preprocessed in the same way:

train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first 25 images from the training set and display the class name below each image.
# Verify that the data is in the correct format and we're ready to build and train the network.

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# The basic building block of a neural network is the layer.
# Layers extract representations from the data fed into them.
# And, hopefully, these representations are more meaningful for the problem at hand.

# Most of deep learning consists of chaining together simple layers.
# Most layers, like tf.keras.layers.Dense, have parameters that are learned during training.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# The first layer in this network, tf.keras.layers.Flatten,
# transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
# Think of this layer as unstacking rows of pixels in the image and lining them up.
# This layer has no parameters to learn; it only reformats the data.

# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.
# These are densely-connected, or fully-connected, neural layers.
# The first Dense layer has 128 nodes (or neurons).
# The second (and last) layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1.
# Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Compile the model
# Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:

# Loss function —This measures how accurate the model is during training.
# We want to minimize this function to "steer" the model in the right direction.
# Optimizer —This is how the model is updated based on the data it sees and its loss function.
# Metrics —Used to monitor the training and testing steps.
# The following example uses accuracy, the fraction of the images that are correctly classified.

model.fit(train_images, train_labels, epochs=5)

# Train the model
# Training the neural network model requires the following steps:

# Feed the training data to the model—in this example, the train_images and train_labels arrays.
# The model learns to associate images and labels.
# We ask the model to make predictions about a test set—in this example, the test_images array.
# We verify that the predictions match the labels from the test_labels array.
# To start training, call the model.fit method—the model is "fit" to the training data.

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i+6000], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i+6000])
    true_label = test_labels[i+6000]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], class_names[true_label]), color=color)
