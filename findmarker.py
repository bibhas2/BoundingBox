#!/usr/local/bin/python

import numpy as np
import tensorflow as tf
from image_loader_flattened import load_image_samples
import matplotlib.image as img

tf.set_random_seed(0)

image = img.imread('yingthai.png')
grey_image = image.mean(2)

width = 225
height = 40

scanIncrement = 2

def constructInput():
	images = []
	classes = []
	filenames = []

	#xOffset = 55
	#yOffset = 500

	for xOffset in xrange(0, grey_image.shape[1] - width, scanIncrement):
		for yOffset in xrange(0, grey_image.shape[0] - height, scanIncrement):

			#this_image = grey_image[0:40 , 0:225]
			#this_image = grey_image[0:225 , 0:40]
			this_image = grey_image[yOffset:(yOffset + height) , xOffset:(xOffset + width)]
			this_image = np.reshape(this_image, this_image.shape[0] * this_image.shape[1])
			images.append(this_image)
			classes.append(np.array([xOffset, yOffset]))
			filenames.append("test")
	
	return np.asarray(images), np.asarray(classes) , filenames

#testImages, testClassification, fileNameList = load_image_samples("test_images")
testImages, testClassification, fileNameList = constructInput()

#Number of classes
K = testClassification[0].shape[0]
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

train_data={X: testImages, Y_: testClassification, W: trainWeights, b: trainBiases}

#Apply the final weights and biases on the training data
checkResult = sess.run(Y, feed_dict=train_data)

maxIndex = 0
maxVal = 0 

for idx, result in enumerate(checkResult):
    #print "File: ", fileNameList[idx]
    #print "Prediction:"
    #print checkResult[idx]
    if checkResult[idx][0] > maxVal:
			maxIndex = idx
			maxVal = checkResult[idx][0]

print "maxIndex: ", maxIndex
print "xOffset: ", testClassification[maxIndex][0]
print "yOffset: ", testClassification[maxIndex][1]
