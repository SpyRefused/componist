# In a regression problem, we aim to predict the output of a continuous value, like a price or a probability.
# Contrast this with a classification problem, where we aim to predict a discrete label
# (for example, where a picture contains an apple or an orange).

# This notebook builds a model to predict the median price of homes in a Boston suburb during the mid-1970s.
# To do this, we'll provide the model with some data points about the suburb, such as the crime rate and
# the local property tax rate.

# This example uses the tf.keras API, see this guide for details.

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# 1) The Boston Housing Prices dataset
# This dataset is accessible directly in TensorFlow. Download and shuffle the training set:

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# 1.1) Examples and features
# This dataset is much smaller than the others we've worked with so far: it has 506 total examples are split
# between 404 training examples and 102 test examples:

print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))  # 102 examples, 13 features

# The dataset contains 13 different features:
#
# 1.- Per capita crime rate.
# 2-. The proportion of residential land zoned for lots over 25,000 square feet.
# 3-. The proportion of non-retail business acres per town.
# 4-. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# 5-. Nitric oxides concentration (parts per 10 million).
# 6-. The average number of rooms per dwelling.
# 7-. The proportion of owner-occupied units built before 1940.
# 8-. Weighted distances to five Boston employment centers.
# 9-. Index of accessibility to radial highways.
# 10-. Full-value property-tax rate per $10,000.
# 11-. Pupil-teacher ratio by town.
# 12-. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
# 13-. Percentage lower status of the population.
#
# Each one of these input data features is stored using a different scale.
# Some features are represented by a proportion between 0 and 1, other features are ranges between 1 and 12,
# some are ranges between 0 and 100, and so on. This is often the case with real-world data, and understanding
# how to explore and clean such data is an important skill to develop.

print(train_data[0])  # Display sample features, notice the different scales

# Use the pandas library to display the first few rows of the dataset in a nicely formatted table:

import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()

# 1.2) Labels
# The labels are the house prices in thousands of dollars. (You may notice the mid-1970s prices.)

print(train_labels[0:10])  # Display first 10 entries

# 2) Normalize features
# It's recommended to normalize features that use different scales and ranges. For each feature,
# subtract the mean of the feature and divide by the standard deviation:

# Test data is *not* used when calculating the mean and std

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized


# Although the model might converge without feature normalization, it makes training more difficult,
# and it makes the resulting model more dependent on the choice of units used in the input.

# 3) Create the model
# Let's build our model. Here, we'll use a Sequential model with two densely connected hidden layers,
# and an output layer that returns a single, continuous value. The model building steps are wrapped in a function,
# build_model, since we'll create a second model, later on.

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model


model = build_model()
model.summary()


# 4) Train the model
# The model is trained for 500 epochs, and record the training and validation accuracy in the history object.

# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

# Visualize the model's training progress using the stats stored in the history object.
# We want to use this data to determine how long to train before the model stops making progress.

import matplotlib.pyplot as plt


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, 5])


plot_history(history)

# This graph shows little improvement in the model after about 200 epochs.
# Let's update the model.fit method to automatically stop training when the validation score doesn't improve.
# We'll use a callback that tests a training condition for every epoch.
# If a set amount of epochs elapses without showing improvement, then automatically stop the training.
#
# You can learn more about this callback here.

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

# The graph shows the average error is about \$2,500 dollars.
# Is this good? Well, $2,500 is not an insignificant amount when some of the labels are only $15,000.

# Let's see how did the model performs on the test set:

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

# 4) Predict
# Finally, predict some housing prices using data in the testing set:

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")

# 5) Conclusion
# This notebook introduced a few techniques to handle a regression problem.
#
# - Mean Squared Error (MSE) is a common loss function
#   used for regression problems (different than classification problems).
# - Similarly, evaluation metrics used for regression differ from classification.
#   A common regression metric is Mean Absolute Error (MAE).
# - When input data features have values with different ranges, each feature should be scaled independently.
# - If there is not much training data, prefer a small network with few hidden layers to avoid overfitting.
# - Early stopping is a useful technique to prevent overfitting.