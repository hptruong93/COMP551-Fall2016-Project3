"""
train_minst.py

This file contains the code which trains a 
model on the original mnist dataset

Note: This file is heavily inspired by:
 https://raw.githubusercontent.com/Lasagne/Lasagne/master/examples/mnist.py

"""
from __future__ import division

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

def errprint(*args):
    sys.stderr.write('\n'.join(map(str,args)) + '\n')

def load_dataset():
    from urllib import urlretrieve
    import gzip

    data_dir = 'data/'

    def require_download(func):
        def wrapper(filename):
            if not os.path.exists(data_dir + filename):
                source = 'http://yann.lecun.com/exdb/mnist/'
                errprint("Downloading {0} to {1}".format(filename, data_dir))
                urlretrieve(source + filename, data_dir + filename)

            return func(data_dir + filename)
        
        return wrapper

    @require_download
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        # reshape into 2D images 
        # shape format: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        return data

    @require_download
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        
        # no reshaping needed for int labels
        return data

    # since we aren't testing our model for just the MNIST
    # digits, we use the test set for validation during training
    # and don't have a test set
    X_train = load_images('train-images-idx3-ubyte.gz')
    y_train = load_labels('train-labels-idx1-ubyte.gz')
    X_valid = load_images('t10k-images-idx3-ubyte.gz')
    y_valid = load_labels('t10k-labels-idx1-ubyte.gz')

    return X_train, y_train, X_valid, y_valid


def build_cnn(input_var):
    network = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input_var)

    # convolutional layer
    # using the settings given in the lasagne example
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=32,
            filter_size=(5,5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # now add a dense layer with dropout 
    #   NOTE: Maybe we don't want dropout (or less dropout), since the 
    #         digits in our data are the same so we might want to overfit
    #         a little bit...
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# iterator for shuffled batches
def batches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for idx in range(0, len(inputs) - batchsize + 1, batchsize):
        batch = indices[idx:idx + batchsize]
        yield inputs[batch], targets[batch]

def train(epochs):
    errprint('Loading data')
    X_train, y_train, X_val, y_val = load_dataset()

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    errprint('Building model and functions')

    network = build_cnn(input_var)
    prediction = lasagne.layers.get_output(network)
    
    # Cross entropy loss function for multiclass network
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Updates for SGD
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss,
            params,
            learning_rate=0.01,
            momentum=0.9)


    valid_prediction = lasagne.layers.get_output(network, deterministic=True)
    valid_loss = lasagne.objectives.categorical_crossentropy(
            valid_prediction,
            target_var).mean()

    valid_acc = T.mean(T.eq(T.argmax(valid_prediction, axis=1), target_var),
            dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [valid_loss, valid_acc])


    errprint('Starting training')

    for epoch in xrange(epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        train_pass = [ 
                train_fn(xs, ys) for xs, ys in batches(X_train, y_train, 500)
        ]
        train_err = sum(train_pass)
        train_batches = len(train_pass)

        valid_pass = [
                val_fn(xs, ys) for xs, ys in batches(X_val, y_val, 500)
        ]

        val_err, val_acc = map(sum,zip(*valid_pass))
        val_batches = len(valid_pass)

        errprint("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, epochs, time.time() - start_time))
        errprint("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        errprint("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        errprint("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))


if __name__ == "__main__":
    train(10)




