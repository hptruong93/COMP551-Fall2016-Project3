"""
logistic_regression.py

COMP 551 Project 3: Team ATP
"""
from __future__ import division

import numpy as np

from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from skimage import measure



def get_central_moments(image, order=5):
    im = image.astype(np.double)
    m = measure.moments(im, order=order)
    cr = m[0, 1] / m[0, 0]
    cc = m[1, 0] / m[0, 0]
    mu = measure.moments_central(im, cr, cc, order=order)
    # flatten array so it can be used as a feature list
    return mu.reshape(-1)

def get_normalized_moments(image, order=5):
    mu = get_central_moments(image, order=order)
    # reshape flattened array to proper matrix
    mu = mu.reshape(order+1, order+1)
    nu = measure.moments_normalized(mu, order=order)
    return np.nan_to_num(nu.reshape(-1))

def get_hu_moments(image, order=None):
    nu = get_normalized_moments(image)
    return np.nan_to_num(nu.reshape(-1))


def get_features(image, method='central', order=5):
    feature_methods = {
            'central': get_central_moments,
            'normalized': get_normalized_moments,
            'hu': get_hu_moments,
            }
    return feature_methods[method](image, order=5)


def train(X, y, method='central', order=5):
    """X should be a (n_images, 60, 60) array, y (n_images) array"""
    X = np.array(map(lambda i: get_features(i, method=method, order=order), X))

    lr = LogisticRegression()
    lr.fit(X, y)
    def predictor(X):
        X = np.array(map(lambda i: get_features(i, method=method, order=order), X))
        return lr.predict(X)
    return predictor

def validate(data):
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    method = data['method']
    order = data['order']
    predict = train(X_train, y_train, method, order)
    results = zip(predict(X_test), y_test)
    acc = len(filter(lambda i: i[0] == i[1], results))
    return acc

def cross_validate(X, y, method='central', order=5):
    fold_cnt = 3
    kf = KFold(len(X), n_folds=fold_cnt)
    folds = [{
        'X_train': X[train],
        'y_train': y[train],
        'X_test': X[test],
        'y_test': y[test],
        'method': method,
        'order': order,
        } for train, test in kf
        ]

    p = Pool(4)

    accs = sum(map(validate, folds)) / len(X)
    print("Accuracy with {} folds: {}".format(fold_cnt, accs))




