import numpy as np
import tensorflow as tf
import tflearn
from image_loader import load_image_samples
from nn_model import build_model

trainImages, trainClassification, fileNames = load_image_samples("train_images")

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

model = build_model(K, imageHeight, imageWidth, numChannels)

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(trainImages, trainClassification, n_epoch=45, show_metric=True)

# Save model when training is complete to a file
print "Saving model"
model.save("marker-classifier.tfl")
print("Network trained and saved as marker-classifier.tfl!")