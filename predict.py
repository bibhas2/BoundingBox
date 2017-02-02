import numpy as np
import tensorflow as tf
import tflearn
from image_loader import load_image_samples
from nn_model import build_model

testImages, testClassification, fileNameList = load_image_samples("test_images")

#Number of classes
K = testClassification.shape[1]
#Number of training samples
m = testImages.shape[0]

#Image dimensions
imageHeight = testImages[0].shape[0]
imageWidth = testImages[0].shape[1]
numChannels = testImages[0].shape[2] #Should be 3 for RGB

print "Number of classes:", K
print "Shape of training samples:", imageHeight, imageWidth, numChannels

model = build_model(K, imageHeight, imageWidth, numChannels)

model.load("./marker-classifier.tfl")

predictedClassification = model.predict(testImages)

for idx, c in enumerate(testClassification):
    print "Expected for: ", fileNameList[idx]
    print c

    print "Predicted"
    print predictedClassification[idx]
    print "======="
