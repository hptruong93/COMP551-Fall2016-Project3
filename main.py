"""COMP 551 (Fall 2016): Project 3

Team ATP
"""
from __future__ import division

import csv

from skimage import filters


import numpy as np
import matplotlib.pyplot as plt

from models.basic_cnn.cnn import train


def view_image(image):
    plt.imshow(image, cmap='Greys_r')
    plt.show()


def load_dataset():
    X = np.fromfile('data/train_X.bin', dtype='uint8')
    X = X.reshape(-1,1,60,60)

    # really simply background removal (for now)
    X = np.apply_along_axis(lambda im: (im > 250).astype(np.float32), 0, X)

    with open('data/train_y.csv','r') as f:
        r = csv.reader(f)
        next(r,None)
        Y = np.array([int(val) for idx, val in r], dtype='uint8')

    # put aside 10k for validation
    X_train, Y_train = X[:-10000], Y[:-10000]
    X_val, Y_val = X[-10000:], Y[-10000:]

    return X_train, Y_train, X_val, Y_val
    



if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_dataset()
    
    train(X_train, y_train, X_val, y_val, 2)



