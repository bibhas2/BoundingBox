import numpy as np
import tensorflow as tf
from image_loader_flattened import load_image_samples

tf.set_random_seed(0)

trainImages, trainClassification, fileNameList = load_image_samples("train_images")

#Number of classes
K = trainClassification[0].shape[0]
#Number of training samples
m = trainImages.shape[0]
#Number of features
n = trainImages.shape[1]

print("Number of classes:", K)
print("Number of training samples:", m)
print("Number of features:", n)

#Training data placeholder
X = tf.placeholder(tf.float32, [m, n])
#Training prediction placeholder
Y_ = tf.placeholder(tf.float32, [m, K])
#Weights
W = tf.Variable(tf.zeros([n, K]))
#Biases
b = tf.Variable(tf.zeros([K]))

# The operation that calculates predictions
Y = tf.nn.softmax(tf.matmul(X, W) + b)

#Cost function
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_graph = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_data={X: trainImages, Y_: trainClassification}

for i in range(1000):
    # train
    sess.run(train_graph, feed_dict=train_data)

    #You can calculate the cost after each iteration if you want.
    #It should steadily decline
    # cost = sess.run(cross_entropy, feed_dict=train_data)
    # print("Cost:", cost)

#Save the weights and biases
finalWeights = sess.run(W)
finalBiases = sess.run(b)

print "Saving weights and biases."
np.save("weights", finalWeights)
np.save("biases", finalBiases)
