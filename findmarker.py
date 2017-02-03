#!/usr/local/bin/python

import numpy as np
import tensorflow as tf
from image_loader_flattened import load_image_samples
import matplotlib.image as img

tf.set_random_seed(0)

#Read the full receipt image
image = img.imread('yingthai.png')
#Convert the image to greyscale
grey_image = image.mean(2)

#Sliding window size is same as the size
#of the marker image we trained with.
windowWidth = 225
windowHeight = 40
strideAmount = 2

def constructInput():
	images = []
	windowCoordinates = []

	for xOffset in xrange(0, grey_image.shape[1] - windowWidth, strideAmount):
		for yOffset in xrange(0, grey_image.shape[0] - windowHeight, strideAmount):

			this_image = grey_image[yOffset:(yOffset + windowHeight) , xOffset:(xOffset + windowWidth)]
			this_image = np.reshape(this_image, this_image.shape[0] * this_image.shape[1])
			images.append(this_image)
			windowCoordinates.append(np.array([xOffset, yOffset]))
	
	return np.asarray(images), np.asarray(windowCoordinates)

testImages, windowCoordinates = constructInput()

#Number of classes
K = 2
#Number of training samples
m = testImages.shape[0]
#Number of features
n = testImages.shape[1]

print("Number of classes:", K)
print("Number of test samples:", m)
print("Number of features:", n)

#Load the trained weights and biases.
trainWeights = np.load("weights.npy")
trainBiases = np.load("biases.npy")

#Training data placeholder
X = tf.placeholder(tf.float32, [m, n])
#Training prediction placeholder
Y_ = tf.placeholder(tf.float32, [m, K])
#Weights
W = tf.placeholder(tf.float32, [n, K])
#Biases
b = tf.placeholder(tf.float32, [K])

# The operation that calculates predictions
Y = tf.nn.softmax(tf.matmul(X, W) + b)

#Cost function
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_data={X: testImages, W: trainWeights, b: trainBiases}

#Apply the final weights and biases on the training data
checkResult = sess.run(Y, feed_dict=train_data)

maxIndex = 0
maxVal = 0 

for idx, result in enumerate(checkResult):
    if checkResult[idx][0] > maxVal:
			maxIndex = idx
			maxVal = checkResult[idx][0]

print "maxIndex: ", maxIndex
print "xOffset: ", windowCoordinates[maxIndex][0]
print "yOffset: ", windowCoordinates[maxIndex][1]
