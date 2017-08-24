import numpy as np
from difflib import SequenceMatcher


class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = {"input": np.random.randn(output_size, input_size),
                        "hidden": np.random.randn(output_size, output_size)}
        self.reset()

    def forward_prop(self, input_v):
        self.data["inputs"].append(input_v)
        output_v = self._forward_prop(input_v)
        self.data["outputs"].append(output_v)
        return output_v

    def _forward_prop(self, input_v):
        pass

    def backwards_prop(self, output_errors, learning_rate):
        amount = len(output_errors)
        errors = self._backwards_prop(output_errors,
                                      [np.zeros_like(output_errors[0])])
        self._apply_adjustments(learning_rate, amount)
        self.reset()
        return errors[:-1]

    def _backwards_prop(self, output_errors, error_is):
        pass

    def _apply_adjustments(self, learning_rate, amount):
        scalar = (learning_rate / amount)
        for weight in self.adjustments:
            self.weights[weight] -= self.adjustments[weight] * scalar

    def reset(self):
        self.adjustments = {"input": np.zeros_like(self.weights["input"])}
        self.data = {"inputs": [], "outputs": []}

    @staticmethod
    def sigmoid(input_v):
        output_v = 1.0 / (1.0 + np.exp(-input_v))
        return output_v

    @staticmethod
    def deriv_sigmoid(input_v):
        output_v = input_v * (1 - input_v)
        return output_v


class OutputLayer(Layer):
    def _forward_prop(self, input_v):
        a = self.weights["input"].dot(input_v)
        return a

    def _backwards_prop(self, output_errors, error_is):
        if not self.data["inputs"]:
            return error_is

        error_k = output_errors.pop()
        input_t = self.data["inputs"].pop()
        error_i = self.weights["input"].T.dot(error_k)
        self.adjustments["input"] += error_k.dot(input_t.T)
        return self._backwards_prop(output_errors, [error_i] + error_is)


class HiddenLayer(Layer):
    def _forward_prop(self, input_v):
        a = self.weights["input"].dot(input_v) +\
            self.weights["hidden"].dot(self.data["outputs"][-1])
        b = self.sigmoid(a)
        return b

    def _backwards_prop(self, output_errors, error_is):
        if not self.data["inputs"]:
            return error_is

        error_k = self.deriv_sigmoid(self.data["outputs"].pop()) * \
            (output_errors.pop() + self.weights["hidden"].T.dot(error_is[-1]))
        input_t = self.data["inputs"].pop()
        error_i = self.weights["input"].T.dot(error_k) * \
            self.deriv_sigmoid(input_t)
        self.adjustments["input"] += error_k.dot(input_t.T)
        self.adjustments["hidden"] += error_k.dot(self.data["outputs"][-1].T)
        return self._backwards_prop(output_errors, [error_i] + error_is)

    def reset(self):
        super().reset()
        self.adjustments["hidden"] = np.zeros_like(self.weights["hidden"])
        self.data["outputs"].append(np.zeros((self.output_size, 1)))


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


def print_network_progress(net, iter_num, training_string, char_list):
    input_vector = np.array([[1]])
    string = ""
    probs = []
    for char in training_string:
        output_v = net.forward_prop(input_vector)
        maxarg = output_v.argmax(axis=0)[0]
        string += str(char_list[maxarg])
        probs.append("{:.3f}".format(output_v[maxarg][0]))

    print(string)
    print(" ".join(probs))
    error = 1 - SequenceMatcher(None, training_string, string).ratio()
    print(error)
    print("--------")
    net.reset()
    return error


total_error = 1
iter_num = 1
net = Network([1, 500, 10])
training_string = "Hello, World!"
char_list = list(set(training_string))
while total_error > 0.01:
    for j in range(1000):
        errors = []
        for char in training_string:
            char_index = char_list.index(char)
            input_vector = np.array([[1]])

            output_v = net.forward_prop(input_vector)
            true_output_v = np.zeros((len(char_list), 1))
            true_output_v[char_index] = 1
            error_k = output_v - true_output_v
            errors.append(error_k)
        net.backwards_prop(errors, learning_rate=min(total_error, 0.0001))

    total_error = print_network_progress(net, iter_num, training_string, char_list)

    iter_num += 1
