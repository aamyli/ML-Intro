# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:35:47 2020

@author: Amy

working through an image classification tutorial, using TensorFlow
focuses on the input + output relationships, in terms of epochs and neuron amounts in the middle layer

"""
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# images, pixel x pixel size
print(train_images.shape)
# num of labels in training set
print(len(train_labels))
# print out the labels (between 0 and 9)
print(train_labels)
# num of images in test set (1/6)
print(test_images.shape)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scale pixels down to range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# show first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# creates a one dimensional neural network
# from the 28x28 images 
# neural network >> data designed to display and 
# seek out patterns, in order to classify them into 
# the various categories

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # this sets the number of NODES in the middle layer, which affects
    # the accuracy of the data as well. the aim is to find a number
    # between the input amount and output amount.
    
    # tested out 128, 64, and 32 neurons with 3 epochs
    # there is a slight drop in accuracy with the decrease of neurons for these numbers
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10)
])

# changes various settings to increase accuracy 
# as model progresses

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# feeding the model
# note that the accuracy increases with each epoch 
# epoch - a pass through the entire dataset

# NOTE: as you change the # of epochs (ie passes through), 
# your final accuracy % AND test % are directly impacted
# ie the more passes = more accurate of a model
model.fit(train_images, train_labels, epochs=3)

# now tests out the test set (10000 images, vs the 60000 training ones)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
# around 88%, less than the trained 91% 
# reaches a point of overfitting - when a ML model performs worse on new data vs on training data

# creates probability sets for the test data
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
# the probability array for the first set 
# distributed as ten numbers, aka the confidence that the 
# image is each of the labels
predictions[0]
# use this to see which label has the highest confidence
np.argmax(predictions[0])

# make plots for each image and their predictions
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# trying the trained model to predict a test image
img = test_images[1]
print(img.shape)
# add to a list (even though it's one image)
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = probability_model.predict(img)
print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
# predicts final value
print(np.argmax(predictions_single[0]))