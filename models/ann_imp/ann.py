import sys
import numpy as np
import csv
import cv2
import time

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class ANN(object):

    ALPHA = 0.5 # Learning rate
    BATCH_SIZE = 1
    BATCH_WEIGHT = 1.0 / BATCH_SIZE

    """
        Raw implementation of artificial neural network.
    """
    def __init__(self, layer_sizes, epochs = 1, activation_function = sigmoid, activation_function_derivative = sigmoid_derivative):
        super(ANN, self).__init__()

        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

        self.number_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        self.epochs = epochs

        # Weight matrix is a 1-index matrix: from 1 to n layers (so 0th element not used)
        # These are the weights of the incoming connections TO this layer.
        # Weight for first layer, which we don't really care since not used,
        self.w = [np.array([])]
        # For all other layers, each element is a matrix (j x k) of j arrays (j nodes at this layer), each has k elements (k nodes at previous layer).
        self.w += [np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) for i in xrange(1, self.number_layers)]

        # Bias matrix is also a 1-index matrix : from 1 to n layers (so 0th element not used)
        # 0th element is the biases for each node at input, which does not exist. Last element is the biases for each node at the last layer.
        self.bias = [np.random.randn(l) for l in layer_sizes]

        # Output matrix is a 1-index matrix representing pre-activation output of the layers from 1 to n (so 0th element is simply the input to the network)
        # 0th element is the input, last element is the output
        self.pre_activation_outputs = [np.zeros(size) for size in self.layer_sizes]

        # Output matrix is a 1-index matrix representing post-activation output of the layers from 1 to n (so 0th element is simply the input to the network)
        # 0th element is the input, last element is the output
        self.layer_outputs = [np.zeros(size) for size in self.layer_sizes]

    def __str__(self):
        return "ANN of {0} layers with sizes {1}".format(self.number_layers, self.layer_sizes)

    def forward_propagation(self, x):
        self.layer_outputs[0] = x

        for i in xrange(1, self.number_layers):
            self.pre_activation_outputs[i] = self.w[i].dot(self.layer_outputs[i - 1]) + self.bias[i]
            self.layer_outputs[i] = self.activation_function(self.pre_activation_outputs[i])

    def backward_propagation(self, y):
        # Assume square error loss function
        # print np.sum(np.square(y - self.layer_outputs[-1]))

        delta_w = [np.zeros(w.shape) for w in self.w]
        delta_bias = [np.zeros(bias.shape) for bias in self.bias]

        # Encode y as 1 hot if possible
        if type(y) is np.uint8:
            one_index = y
            y = np.zeros((19,), dtype = np.uint8)
            y[one_index] = 1

        assert y.shape == self.layer_outputs[-1].shape

        # First we need to calculate the derivative at the output layer
        error = -(y - self.layer_outputs[-1]) * self.activation_function_derivative(self.pre_activation_outputs[-1])

        # self.bias[-1] -= self.ALPHA * error
        delta_bias[-1] = self.ALPHA * error

        error = error.reshape(-1,1)
        # self.w[-1] -= self.ALPHA * error.dot(self.layer_outputs[-2].reshape(-1,1).T)
        delta_w[-1] = self.ALPHA * error.dot(self.layer_outputs[-2].reshape(-1,1).T)

        for layer in xrange(self.number_layers - 2, 0, -1):
            error = self.w[layer + 1].T.dot(error) * self.activation_function_derivative(self.layer_outputs[layer].reshape(-1, 1))

            assert error.flatten().shape == self.bias[layer].shape
            # self.bias[layer] -= self.ALPHA * error.flatten()
            delta_bias[layer] = self.ALPHA * error.flatten()

            delta = error.dot(self.layer_outputs[layer-1].reshape(-1,1).T)
            assert self.w[layer].shape == delta.shape
            # self.w[layer] -= self.ALPHA * delta
            delta_w[layer] = self.ALPHA * delta

        return delta_w, delta_bias

    def fit(self, X, y):
        delta_w = [np.zeros(w.shape) for w in self.w]
        delta_bias = [np.zeros(bias.shape) for bias in self.bias]

        count = 0

        for epoch in xrange(self.epochs):
            print("Epoch {0}".format(epoch))
            for i, x in enumerate(X):
                if i % 1000 == 0:
                    print("Iteration {0}".format(i))

                if count == self.BATCH_SIZE:
                    count = 0
                    for i in xrange(len(self.w)):
                        self.w[i] -= delta_w[i]
                        self.bias[i] -= delta_bias[i]

                    delta_w = [np.zeros(w.shape) for w in self.w]
                    delta_bias = [np.zeros(bias.shape) for bias in self.bias]

                count += 1

                self.forward_propagation(x)
                new_delta_w, new_delta_bias = self.backward_propagation(y[i])

                new_delta_w = [self.BATCH_WEIGHT * dw for dw in new_delta_w]
                new_delta_bias = [self.BATCH_WEIGHT * db for db in new_delta_bias]
                for i in xrange(self.number_layers):
                    delta_w[i] += new_delta_w[i]
                    delta_bias[i] += new_delta_bias[i]


    def predict_single(self, x, classify = False):
        self.forward_propagation(x)
        return np.argmax([self.layer_outputs[-1]]) if classify else self.layer_outputs[-1]

    def predict(self, X, classify = False):
        return [self.predict_single(x, classify) for x in X]

    def score(self, X, y):
        # Assume sum square error loss
        predicted = self.predict(X)
        return np.sum(np.square(np.array([predicted[i] - true_val for i, true_val in enumerate(y)])))

################################################################################################
def preprocess(X):
    # really simply background removal (for now)
    X = np.apply_along_axis(lambda im: (im > 252).astype(np.float32), 0, X)

    def crop(im):
        return im[2:-2, 2:-2]

    def resize(im):
        # return cv2.resize(im, (0,0), fx = 0.5, fy = 0.5)
        return im

    X = np.array([resize(crop(im)) for im in X])
    X = X.reshape((-1, (28*2) **2))

    return X

def load_images():
    print("....Loading training and validation images")
    X = np.fromfile('../../data/train_x.bin', dtype='uint8')
    X = X.reshape((100000,60,60))

    # X = X[:500]

    print("....Preprocess training images")
    X = preprocess(X)
    return X

def load_labels():
    print("....Loading labels")
    with open('../../data/train_y.csv','r') as f:
        r = csv.reader(f)
        next(r,None)
        Y = np.array([int(val) for idx, val in r], dtype='uint8')
    return Y

def load_test():
    print("....Loading test images")
    test = np.fromfile('../../data/test_x.bin', dtype='uint8')
    test = test.reshape((20000,60,60))

    print("....Preprocess test images")
    test = preprocess(test)
    return test

def load_all_data():
    images = load_images()
    labels = load_labels()
    tests = load_test()

    total = len(images)
    validation_size = int(total * 0.2)
    print("Reserving {0} for validation set.".format(validation_size))

    train_x, val_x = images[:-validation_size], images[-validation_size:]
    train_y, val_y = labels[:-validation_size], labels[-validation_size:]
    test_x = tests

    return ((train_x, train_y), (val_x, val_y), (test_x))

def write_output(predictions):
    ids = list(xrange(20000))

    with open('../data/predict_lenet_epoch_2000.csv', 'w') as f:
        np.savetxt(f, np.c_[ids, predictions], header = 'Id,Prediction', delimiter = ',', fmt='%s', comments = '')

################################################################################################

if __name__ == "__main__":
    epoch = int(sys.argv[1])
    sizes = [int(arg) for arg in sys.argv[2:]]
    sizes = [56*56] + sizes + [19]

    np.random.seed(int(time.time()))
    loaded_data = load_all_data()

    train, val, test = loaded_data
    test = (test, np.array(tuple(np.random.randint(0, 20) for _ in xrange(20000))))

    ann = ANN(sizes, epochs = 2)
    print("Constructed ANN {0}".format(ann))

    start_time = time.time()
    ann.fit(train[0], train[1])
    print("Training took {0}m".format((time.time() - start_time) / 60.0))

    validation_predictions = ann.predict(val[0], classify = True)
    accuracy = np.sum([1 if v == val[1][i] else 0 for i, v in enumerate(validation_predictions)]) / float(len(val[0]))
    print("Validation Accuracy is {0}".format(accuracy))
    print("Finished running for ANN {0}".format(ann))

    # predictions = ann.predict(test[0], classify = True)
    # print(predictions)
