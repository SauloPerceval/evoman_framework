import numpy

input_size = 20
hidden_layer_neurons = 10
output_size = 4


class NeuralNetwork:

    def __init__(self, weights=None):
        weights = (numpy.random.normal(loc=0.5,
                                       scale=0.1,
                                       size=input_size*hidden_layer_neurons+hidden_layer_neurons*output_size)
                   if not weights else weights)
        self.first_weights = weights[:input_size*hidden_layer_neurons].reshape(hidden_layer_neurons, input_size)
        self.second_weights = weights[-hidden_layer_neurons*output_size:].reshape(output_size, hidden_layer_neurons)

    def _first_layer(self, inputs):
        apply_weights = self.first_weights * inputs
        # Applies ReLu activation function
        first_layer_output = numpy.array(list(map(lambda x: numpy.maximum(0, sum(x)), apply_weights)))

        return first_layer_output

    def _second_layer(self, layer_inputs):
        apply_weights = self.second_weights * layer_inputs
        # Applies tanh activation function
        second_layer_output = numpy.array(list(map(lambda x: numpy.tanh(sum(x)), apply_weights)))

        return second_layer_output

    def run_network(self, inputs):
        first_layer_output = self._first_layer(inputs)
        second_layer_output = self._second_layer(first_layer_output)

        return second_layer_output

    def weights_to_1D_array(self):
        return numpy.concatenate((self.first_weights.flatten(), self.second_weights.flatten()))
