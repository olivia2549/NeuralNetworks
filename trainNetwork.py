import numpy as np
import matplotlib.pyplot as plt

# supress some unnecessary warnings
import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load mnist images (60,000 train images and 10,000 test images; each image is 28x28 pixels gray scale)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# number of images
Ntr = train_images.shape[0]  # 60000
Nts = test_images.shape[0]  # 10000

# image shape
szx = train_images.shape[1]  # 28
szy = train_images.shape[2]  # 28

# need to reshape the 28x28 training/testing images as vectors
train_images_vec = train_images.reshape((Ntr, szx * szy))
test_images_vec = test_images.reshape((Nts, szx * szy))

# deciding to normalize the pixels to 0..1 and recase as float32
train_images_vec = train_images_vec.astype('float32') / 255
test_images_vec = test_images_vec.astype('float32') / 255

# also need to categorically encode the labels as "one hot"
train_labels_onehot = to_categorical(train_labels)
test_labels_onehot = to_categorical(test_labels)

# define and train neural network
nout = 10  # number of units in the output (activation) layer

# create architecture of simple neural network model
# input layer  : 28*28 = 784 input nodes
# output layer : 10 (nout) output nodes
network = models.Sequential()
network.add(layers.Dense(nout, activation='sigmoid', input_shape=(szx * szy,)))

# print a model summary
print(network.summary())
print()
for layer in network.layers:
    print('layer name : {} | input shape : {} | output shape : {}'.format(layer.name, layer.input.shape,
                                                                          layer.output.shape))
print()
for layer in network.layers:
    print(layer.get_config())
print()

# compile network
network.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

# now train the network
history = network.fit(train_images_vec, train_labels_onehot, verbose=True, validation_split=.1, epochs=20,
                      batch_size=128)
print('Done training!')
print()

# test network
test_loss, test_acc = network.evaluate(test_images_vec, test_labels_onehot, verbose=True)
print('test_acc:', test_acc)

# get learned network weights and biases

# the weight matrix is 784 by 10 because there's 784 input neurons and 10 output neurons
# bias is just 10 numbers, one for each output neuron
W = network.layers[0].get_weights()[0]  # weights input to hidden
B = network.layers[0].get_weights()[1]  # bias to hidden

# model predictions (all 10000 test images)
out = network.predict(test_images_vec)


def one_hot_to_labels(output):
    test_decisions = np.zeros(len(test_labels))
    for i in range(len(output)):
        winning_activation = 0
        for j in range(len(output[i])):
            if output[i][j] > winning_activation:
                winning_activation = output[i][j]
                test_decisions[i] = int(j)
    return test_decisions


# fill out the correct and incorrect numbers (starting from a random test_labels location)
def find_correct_and_incorrect(s, labels, decisions, num_outputs):
    correct_i = []
    correct_n = []
    incorrect_i = []
    incorrect_n = []
    while len(correct_i) < num_outputs or len(incorrect_i) < num_outputs:
        if labels[s] not in correct_n and labels[s] == int(decisions[s]):
            correct_i.append(s)
            correct_n.append(labels[s])
        if labels[s] not in incorrect_n and labels[s] != int(decisions[s]):
            incorrect_i.append(s)
            incorrect_n.append(labels[s])
        s += 1  # increment i
    return correct_i, incorrect_i


# given a list of correct and incorrect labels, show the corresponding images and answer given
def plot_correct_and_incorrect(correct_i, incorrect_i, num_outputs):
    for i in range(2 * num_outputs):
        plt.subplot(2, num_outputs, i + 1)
        idx = 0
        if i < num_outputs:
            idx = correct_i[i]
        else:
            idx = incorrect_i[i - num_outputs]
        plt.imshow(test_images[idx], cmap='gray', interpolation='none')
        plt.title("{}/{}".format(test_labels[idx], int(test_decisions[idx])))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# compare 2D arrays for differences
def check_arrays(array, correct_array, epsilon):
    diff = array - correct_array
    # find all occurrences of differences greater than epsilon among the rows
    diffs_rows = np.where(np.abs(diff) > epsilon)[0]
    # find all occurrences of differences greater than epsilon among the columns
    diffs_cols = np.where(np.abs(diff) > epsilon)[1]
    if diffs_rows.size == 0 and diffs_cols.size == 0:
        print("no differences found")
    else:
        print("fail")


# apply the sigmoid activation function (from HW2)
def logistic_func_vectorized(n):
    return 1 / (1 + np.exp(-n))


test_decisions = one_hot_to_labels(out)
(correct_indices, incorrect_indices) = find_correct_and_incorrect(80, test_labels, test_decisions, len(W[0]))
plot_correct_and_incorrect(correct_indices, incorrect_indices, len(W[0]))

# initialize outputs array with correct shape
outputs = np.empty((len(test_labels), len(W[0])))

for i in range(len(test_labels)):  # for each test image input
    inputs = test_images_vec[i].reshape(1, len(W[:, 0]))  # make shape compatible
    outputs[i] = logistic_func_vectorized(np.dot(inputs, W) + B)  # apply activation func

# check with the values produced by Keras (using epsilon of 0.0001)
check_arrays(outputs, out, 0.0001)
