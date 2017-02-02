import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.conv import conv_2d, max_pool_2d

def build_model(K, imageHeight, imageWidth, numChannels):
    tf.reset_default_graph()
    
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
    #model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='marker-classifier.tfl.ckpt')
    model = tflearn.DNN(network)

    return model