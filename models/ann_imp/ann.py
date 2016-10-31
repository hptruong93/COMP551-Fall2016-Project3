import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class ANN(object):

    ALPHA = 0.05 # Learning rate

    """
        Raw implementation of artificial neural network.
    """
    def __init__(self, layer_sizes, activation_function = sigmoid, activation_function_derivative = sigmoid_derivative):
        super(ANN, self).__init__()

        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

        self.number_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

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
        print "AAA ", map(lambda x : x.shape, self.w)
        print "AAA ", map(lambda x : x.shape, self.bias)
        print "BBB ", map(lambda x : x.shape, self.pre_activation_outputs)
        print "CCC ", map(lambda x : x.shape, self.layer_outputs)

    def forward_propagation(self, x):
        self.layer_outputs[0] = x

        for i in xrange(1, self.number_layers):
            self.pre_activation_outputs[i] = self.w[i].dot(self.layer_outputs[i - 1]) + self.bias[i]
            self.layer_outputs[i] = self.activation_function(self.pre_activation_outputs[i])

    def backward_propagation(self, y):
        # Assume square error loss function
        print np.sum(np.square(y - self.layer_outputs[-1]))

        # First we need to calculate the derivative at the output layer
        error = -(y - self.layer_outputs[-1]) * self.activation_function_derivative(self.pre_activation_outputs[-1])

        self.bias[-1] -= self.ALPHA * error
        error = error.reshape(-1,1)
        self.w[-1] -= self.ALPHA * error.dot(self.layer_outputs[-2].reshape(-1,1).T)

        for layer in xrange(self.number_layers - 2, 0, -1):
            error = self.w[layer + 1].T.dot(error) * self.activation_function_derivative(self.layer_outputs[layer].reshape(-1, 1))



            # print self.w[layer].shape, error.shape, self.bias[layer].shape
            # print "WHAT " , (self.ALPHA * error).shape
            # print self.layer_outputs[layer-1].reshape(-1,1).shape

            assert error.flatten().shape == self.bias[layer].shape
            self.bias[layer] -= self.ALPHA * error.flatten()

            delta_w = error.dot(self.layer_outputs[layer-1].reshape(-1,1).T)
            assert self.w[layer].shape == delta_w.shape
            self.w[layer] -= self.ALPHA * delta_w



if __name__ == "__main__":
    ann = ANN([2,3,4,2])

    for i in xrange(100):
        ann.forward_propagation(np.array([i / 100.0, 0.01 + i / 100.0]))
        ann.backward_propagation([i / 100.0, 0.01 + i / 100.0])