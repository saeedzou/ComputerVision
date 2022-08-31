import math
import numpy as np
import h5py
import tensorflow as tf


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def LeNet5(input_shape, output_shape):
    """

        Args:
            output_shape: number of output classes
            input_shape: shape of input image without batch size e.g) (64, 64, 3)

        Returns:
            model: modified AlexNet based on the LeNet5 paper by Yann Lecun

        """
    i = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(6, (5, 5), strides=1, activation='tanh')(i)
    x = tf.keras.layers.AveragePooling2D(2, strides=2)(x)
    x = tf.keras.layers.Conv2D(16, (5, 5), strides=1, activation='tanh')(x)
    x = tf.keras.layers.AveragePooling2D(2, strides=2)(x)
    x = tf.keras.layers.Conv2D(120, (5, 5), strides=1, activation='tanh')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(120, activation='tanh')(x)
    x = tf.keras.layers.Dense(84, activation='tanh')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.models.Model(i, x)
    return model


def AlexNet(input_shape, output_shape):
    """

    Args:
        output_shape: number of output classes
        input_shape: shape of input image without batch size e.g) (64, 64, 3)

    Returns:
        model: AlexNet architecture

    """
    i = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu')(i)
    x = tf.keras.layers.MaxPool2D((3, 3), strides=2)(x)
    x = tf.keras.layers.Conv2D(256, (5, 5), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D((3, 3), strides=2)(x)
    x = tf.keras.layers.Conv2D(384, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(384, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.models.Model(i, x)
    return model


