import numpy as np
import random
from time import sleep

input_layer = 2
hidden_layer = 10
output_layer = 1
hidden_weights = np.random.random((hidden_layer, input_layer))
output_weights = np.random.random((output_layer, hidden_layer))
a_k = None
b_h = None
b_i = None


def sigmoid(input_v):
    output_v = 1.0 / (1.0 + np.exp(-input_v))
    return output_v


def deriv_sigmoid(input_v):
    output_v = input_v * (1 - input_v)
    return output_v


def forward_prop(input_v):
    global hidden_weights
    global output_weights
    global a_k
    global b_h
    global b_i
    b_i = input_v
    a_h = hidden_weights.dot(b_i)
    b_h = sigmoid(a_h)
    a_k = output_weights.dot(b_h)
    return a_k


def backwards_prop(error_k, learning_rate=0.001):
    global hidden_weights
    global output_weights
    global a_k
    global b_h
    global b_i
    error_h = output_weights.T.dot(error_k) * deriv_sigmoid(b_h)
    hidden_weights -= error_h.dot(b_i.T) * learning_rate
    output_weights -= error_k.dot(b_h.T) * learning_rate

total_error = 1
while total_error > 0.001:
    for j in range(10000):
        a = random.randint(0, 1)
        b = random.randint(0, 1)
        input_vector = np.array([[a], [b]])
        output_v = forward_prop(input_vector)
        error_k = output_v - np.array([[a and b]])
        error_value = np.sum(error_k**2)
        backwards_prop(error_k, learning_rate=total_error/10)

    total_error = 0
    for j in range(4):
        a = j & 1
        b = (j >> 1) & 1
        input_vector = np.array([[a], [b]])
        output_v = forward_prop(input_vector)
        local_error = np.abs(output_v - np.array([[a and b]]))[0][0]
        total_error += local_error
        print("{} && {} = {:.6f} \tError: {:.6f}".format(a, b, output_v[0][0], local_error))
    print("Total Error: {:.6f}".format(total_error))
    print("---------")
