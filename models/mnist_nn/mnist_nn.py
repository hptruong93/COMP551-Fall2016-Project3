"""
mnist_nn.py

This file contains the code which trains a 
uses the results of a MNIST trained CNN on 
28x28 subimages of an image as features in
a MLP NN

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

from models.mnist.train_mnist import load_predictor

mnist_predictor = load_predictor()

#NOTE: This should be a power of two less than 32
SUBIMAGE_STEP = 4

dir_path = os.path.dirname(os.path.realpath(__file__))
model_file = os.path.join(dir_path, 'mnist_nn_model.npz')

def errprint(*args):
    sys.stderr.write('\n'.join(map(str,args)) + '\n')


def extract_features(images, step):
    def get_subimages(image):
        # grab all the 28x28 subimages, stepping along by 2 pixels
        #   32 becaause 32 = 60 - 28
        #   this creates 32*32/(step**2) subimages
        sub_images = np.array(
                [image[i:i+28,j:j+28] 
                    for i in range(0,32,step) for j in range(0,32,step)]
        )
        return sub_images
    
    # number of subimages made per image
    num_si = 32**2 // step**2

    images = images.reshape(len(images),60,60)
    sub_images = map(get_subimages, images)

    # apply the predictor to each subimage
    # do it in batches of 500 to avoid blowing up memory
    flatten = lambda l: [i for sl in l for i in sl]
    shaper = lambda l: np.array(l).reshape(num_si*len(l), 1, 28, 28)

    results = flatten(map(mnist_predictor, 
        [shaper(sub_images[i*500:(i+1)*500]) for i in xrange(len(sub_images)//500)]
        ))

    # reshape to group predictions by sub image for original image
    results = np.array(results).reshape(len(images), 1, num_si, 10)
    
    # results is now an array of sum_si x 10 where the 
    # num_si is each subimage and the 10 is the confidence 
    # for a character for that
    return results


def build_nn(input_var, num_subimages):
    # input layer takes num_si rows of confidences for each possible digit
    network = lasagne.layers.InputLayer(
            shape=(None, 1, num_subimages, 10),
            input_var=input_var)

    # three dense layers of 800 with a dropout in the middle
    network = lasagne.layers.DenseLayer(
            network, 
            num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform()) 

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform()) 

    network = lasagne.layers.DenseLayer(
            network, 
            num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform()) 



    # now add a output layer layer with dropout 
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
    for idx in range(0, len(inputs) - batchsize + 1, batchsize):
        batch = indices[idx:idx + batchsize]
        yield inputs[batch], targets[batch]

def train(X_train, y_train, X_val, y_val, epochs):
    # Takes X_train and X_val in the format of an array of
    # arrays 64 rows of 10, where each row is a subimage and 
    # each column is a prediction for a digit of that subimage
    # shape: (len(X), 1, 64, 10)
    if os.path.exists(model_file):
        errprint("Model file exists. Please delete it and run again")
        return

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    errprint('Building model and functions')

    network = build_nn(input_var, 32**2 // SUBIMAGE_STEP**2)
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


