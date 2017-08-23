import numpy as np
import random
import pdb
from time import sleep

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.random((output_size, input_size))
        self.prev_inputs = []

    def forward_prop(self, input_v):
        return input_v

    def backwards_prop(self, output_errors, learning_rate):
        adjustment_matrix = np.zeros_like(self.weights)
        return self._backwards_prop(output_errors, learning_rate, adjustment_matrix, 0)

    def _backwards_prop(self, output_errors, learning_rate, adjustment_matrix, depth):
        if not self.prev_inputs:
            self.weights -= adjustment_matrix * (learning_rate / depth)
            return []
        error_k = output_errors.pop()
        input_t = self.prev_inputs.pop()
        error_i = self.weights.T.dot(error_k) * self.deriv_sigmoid(input_t)
        adjustment_matrix += error_k.dot(input_t.T)
        return self._backwards_prop(output_errors, learning_rate, adjustment_matrix, depth + 1) + [error_i]

    def reset(self):
        self.prev_inputs = []

    @staticmethod
    def sigmoid(input_v):
        output_v = 1.0 / (1.0 + np.exp(-input_v))
        return output_v

    @staticmethod
    def deriv_sigmoid(input_v):
        output_v = input_v * (1 - input_v)
        return output_v

class OutputLayer(Layer):
    def forward_prop(self, input_v):
        self.prev_inputs.append(input_v)
        a = self.weights.dot(input_v)
        return a

class HiddenLayer(Layer):

    def forward_prop(self, input_v):
        self.prev_inputs.append(input_v)
        a = self.weights.dot(input_v)
        b = self.sigmoid(a)
        return b

class Network:
    def __init__(self, layer_sizes):
        self.hidden_layer = HiddenLayer(layer_sizes[0], layer_sizes[1])
        self.output_layer = OutputLayer(layer_sizes[1], layer_sizes[2])

    def forward_prop(self, input_v):
        out = self.hidden_layer.forward_prop(input_v)
        return self.output_layer.forward_prop(out)

    def backwards_prop(self, error_k, learning_rate):
        error = error_k
        error = self.output_layer.backwards_prop(error, learning_rate)
        error = self.hidden_layer.backwards_prop(error, learning_rate)
        return error

    def reset(self):
        self.hidden_layer.reset()
        self.output_layer.reset()

def print_network_progress(net, iter_num):
    total_error = 0
    print("Iter: {}".format(iter_num))
    for j in range(4):
        a = j & 1
        b = (j >> 1) & 1
        input_vector = np.array([[a], [b]])
        output_v = net.forward_prop(input_vector)
        local_error = np.abs(output_v - np.array([[a and b]]))[0][0]
        total_error += local_error
        print("{} && {} = {:.6f} \tError: {:.6f}".format(a, b, output_v[0][0], local_error))
    print("Total Error: {:.6f}".format(total_error))
    print("---------")
    net.reset()
    return total_error

total_error = 1
iter_num = 1
net = Network([2, 10, 1])
while total_error > 0.01:
    for j in range(10000):
        errors = []
        for i in range(10):
            a = random.randint(0, 1)
            b = random.randint(0, 1)
            input_vector = np.array([[a], [b]])

            output_v = net.forward_prop(input_vector)
            error_k = output_v - np.array([[a and b]])
            errors.append(error_k)
        net.backwards_prop(errors, learning_rate=total_error/10)

        error_value = np.sum(error_k**2)

    total_error = print_network_progress(net, iter_num)

    iter_num += 1
