import numpy as np
import random
from time import sleep

input_layer = 2
hidden_layer = 5
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


def backwards_prop(error_k, learning_rate=0.1):
    global hidden_weights
    global output_weights
    global a_k
    global b_h
    global b_i
    error_h = output_weights.T.dot(error_k) * deriv_sigmoid(b_h)
    hidden_weights -= error_h.dot(b_i.T) * learning_rate
    output_weights -= error_k.dot(b_h.T) * learning_rate


for i in range(10):
    for j in range(10000):
        a = random.randint(0, 2)
        b = random.randint(0, 2)
        input_vector = np.array([[a], [b]])
        output_v = forward_prop(input_vector)
        error_k = output_v - np.array([[a and b]])
        backwards_prop(error_k)
    for j in range(4):
        a = j & 1
        b = (j >> 1) & 1
        input_vector = np.array([[a], [b]])
        output_v = forward_prop(input_vector)
        print("{} && {} = {}".format(a, b, output_v[0][0]))
    print("---------")
