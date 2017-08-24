import numpy as np


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


def print_network_progress(net, iter_num):
    total_error = 0
    print("Iter: {}".format(iter_num))
    state = 0
    states = [0, 1, 0]
    for j in range(len(states)):
        a = states.pop()
        input_vector = np.array([[a]])
        output_v = net.forward_prop(input_vector)
        state ^= a
        local_error = np.abs(output_v - np.array([[state]]))[0][0]
        total_error += local_error
        print("state ^= {:d} = {:.3f} ({}) \tError: {:.3f}"
              .format(a, output_v[0][0], state, local_error))
    print("Total Error: {:.3f}".format(total_error))
    print("---------")
    net.reset()
    return total_error


total_error = 1
iter_num = 1
net = Network([1, 10, 1])
while total_error > 0.01:
    for j in range(1000):
        errors = []
        state = 0
        for i in range(3):
            a = np.random.randint(0, 2)
            input_vector = np.array([[a]])

            output_v = net.forward_prop(input_vector)
            state ^= a
            error_k = output_v - np.array([[state]])
            errors.append(error_k)
        net.backwards_prop(errors, learning_rate=min(total_error, 0.0001))

        error_value = np.sum(error_k**2)

    total_error = print_network_progress(net, iter_num)

    iter_num += 1
