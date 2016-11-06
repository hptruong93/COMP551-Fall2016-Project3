"""
cnn.py

This file contains the code which trains a 
Convolutional NN on the full 60x60 pixel
images of our data set.

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


dir_path = os.path.dirname(os.path.realpath(__file__))
model_file = os.path.join(dir_path, 'basic_cnn_model.npz')

def errprint(*args):
    print('\n'.join(map(str,args)))
    sys.stderr.write('\n'.join(map(str,args)) + '\n')

def build_cnn(input_var):
    network = lasagne.layers.InputLayer(
            shape=(None, 1, 60, 60),
            input_var=input_var)

    # 5 convolutional layers
    # using the settings given in the lasagne example
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=64,
            filter_size=(5,5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())


    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=64,
            filter_size=(3,3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=64,
            filter_size=(3,3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=64,
            filter_size=(3,3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=64,
            filter_size=(3,3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)


    # More convolutional layers
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=96,
            filter_size=(3,3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=96,
            filter_size=(3,3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)


    # now add a dense layer with dropout 
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=19,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# iterator for shuffled batches
def batches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    total = len(inputs) // batchsize
    for i, idx in enumerate(range(0, len(inputs) - batchsize + 1, batchsize)):
        batch = indices[idx:idx + batchsize]
        # sys.stderr.write("Batch {}/{}\n".format(str(i), str(total)))
        yield inputs[batch], targets[batch]

def train(X_train, y_train, X_val, y_val, epochs):
    if os.path.exists(model_file):
        errprint("Model file exists. Please delete it and run again")
        return

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
            learning_rate=0.001,
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

    # write intermediate param values to temporary file
    tmp_model_file = model_file[:-4] + '.tmp.npz'
    errprint("Writing intermediate params to: %s" % tmp_model_file)

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

        errprint("Updating intermediate params file")
        if os.path.exists(tmp_model_file):
            os.remove(tmp_model_file)
        np.savez(tmp_model_file, *lasagne.layers.get_all_param_values(network))
        

    errprint("Finished training. Saving results to file")

    np.savez(model_file, *lasagne.layers.get_all_param_values(network))
    os.remove(tmp_model_file)
    return input_var, network


def load_predictor():
    assert os.path.exists(model_file), "Must train model to load predictor"
        
    input_var = T.tensor4('inputs')
    network = build_cnn(input_var)
    with np.load(model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    
    prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fcn = theano.function([input_var], prediction)

    def predictor(images):
        """
Takes an np array of 60x60 normalized greyscale images (shape (1, 60, 60) )
and returns the top prediction and the confidence of the prediction
        """

        def get_result(confs):
            argmax = np.argmax(confs)
            return (argmax, confs[argmax])

        results = map(get_result, predict_fcn(images))
        return results

    return predictor


if __name__ == "__main__":
    epochs = 200 if len(sys.argv) < 2 else int(sys.argv[1])
    train(epochs)


