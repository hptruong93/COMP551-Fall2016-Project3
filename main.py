"""COMP 551 (Fall 2016): Project 3

Team ATP
"""
from __future__ import division

import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

from models.basic_cnn.cnn import train as cnn_train
from models.logistic_regression.logistic_regression import *

def view_image(image):
    plt.imshow(image, cmap='Greys_r')
    plt.show()


def load_images():
    X = np.fromfile('data/train_x.bin', dtype='uint8')
    X = X.reshape(-1,1,60,60)

    # really simply background removal (for now)
    X = np.apply_along_axis(lambda im: (im > 252).astype(np.float32), 0, X)
    return X

def load_labels():
    with open('data/train_y.csv','r') as f:
        r = csv.reader(f)
        next(r,None)
        Y = np.array([int(val) for idx, val in r], dtype='uint8')
    return Y

def load_dataset():
    X = load_images()
    Y = load_labels()

    # put aside 10k for validation
    X_train, Y_train = X[:-10000], Y[:-10000]
    X_val, Y_val = X[-10000:], Y[-10000:]

    return X_train, Y_train, X_val, Y_val


def load_mnist_prediction_dataset():
    print("Loading training subimage predictions")
    X_train = np.loadtxt('data/mnist_predictions_train_8stp.txt')
    X_train = X_train.reshape(-1, 1, 16, 10)

    print("Loading validation subimage predictions")
    X_val = np.loadtxt('data/mnist_predictions_val_8stp.txt')
    X_val = X_val.reshape(-1, 1, 16, 10)

    Y = load_labels()
    Y_train, Y_val = Y[:-10000], Y[-10000:]

    return X_train, Y_train, X_val, Y_val

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_dataset()

    cnn_train(X_train, y_train, X_val, y_val, 100)

#    X = np.append(X_train, X_val)
#    X = X.reshape(-1, 60, 60)
#    y = np.append(y_train, y_val)
#
#    for order in [5, 10, 15]:
#        for method in ['central', 'normalized', 'hu']:
#            print("Cross validation for order {} and method {}".format(order, method))
#            cross_validate(X[:20000], y[:20000], method=method, order=order)
#


