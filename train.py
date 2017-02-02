import numpy as np
import tensorflow as tf
import tflearn
from image_loader import load_image_samples
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.conv import conv_2d, max_pool_2d

trainImages, trainClassification = load_image_samples("train_images")

# for img in trainImages:
#     print img.shape

#Number of classes
K = trainClassification.shape[1]
#Number of training samples
m = trainImages.shape[0]

#Image dimensions
imageHeight = trainImages[0].shape[0]
imageWidth = trainImages[0].shape[1]
numChannels = trainImages[0].shape[2] #Should be 3 for RGB

print "Number of classes:", K
print "Shape of training samples:", imageHeight, imageWidth, numChannels

#1 - Input layer
network = input_data(shape=[None, imageHeight, imageWidth, numChannels])

#2 - Conv layer
network = conv_2d(network, 4, 5, activation='relu')

#3 - Max pooling
network = max_pool_2d(network, 2)

#4 - Conv layer
network = conv_2d(network, 4, 5, activation='relu')

#5 - Max pooling
network = max_pool_2d(network, 2)

#6 Fully connected layer. Let's use 200 neurons
network = fully_connected(network, 200, activation='relu')

#7 Final step. Readout layer. K neurons, one per class.
#Final layer uses softmax activation.
network = fully_connected(network, K, activation='softmax')

#The cost function
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='marker-classifier.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(trainImages, trainClassification, n_epoch=35, show_metric=True)
# model.fit(trainImages, trainClassification, n_epoch=100, shuffle=True, 
#           show_metric=True, batch_size=m,
#           snapshot_epoch=True,
#           run_id='marker-classifier')

# Save model when training is complete to a file
model.save("marker-classifier.tfl")
print("Network trained and saved as marker-classifier.tfl!")