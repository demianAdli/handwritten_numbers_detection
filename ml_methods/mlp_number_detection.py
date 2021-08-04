import numpy as np
import scipy.special as spy


class NumDetectionMLP:
    def __init__(self, input_nodes_num, hidden_nodes_num,
                 output_nodes_num, learning_rate):

        self.input_nodes_num = input_nodes_num
        self.hidden_nodes_num = hidden_nodes_num
        self.output_nodes_num = output_nodes_num
        self.learning_rate = learning_rate

        # matrix for links' weights from input to hidden layer
        self.input_weights = np.random.\
            normal(0, pow(self.hidden_nodes_num, -0.5),
                   (self.hidden_nodes_num, self.input_nodes_num))

        # matrix for links' weights from hidden to output layer
        self.hidden_weights = np.random.\
            normal(0, pow(self.output_nodes_num, -0.5),
                   (self.output_nodes_num, self.hidden_nodes_num))

        # calling the sigmoid function for the activation function
        self.activation_function = lambda x: spy.expit(x)

    def train(self, inputs_list, targets_list):
        # converts inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculates signals based on the weights of input links
        # into hidden layer
        hidden_inputs = np.dot(self.input_weights, inputs)

        # Hidden nodes are the first nodes with
        # an activation (sigmoid) function.
        # New weights are computed based on the sigmoid
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculates signals based on the weights on the
        # links from the hidden layer
        final_inputs = np.dot(self.hidden_weights, hidden_outputs)

        # Each of output layer's nodes also include s
        # an activation (sigmoid) function.
        # New weights are computed based on the sigmoid
        final_outputs = self.activation_function(final_inputs)

        # the model's error
        output_errors = targets - final_outputs

        # Hidden layer error is copmuted based on
        # applying of the model's error on the weights
        # from the output layer to the hidden layer (Backpropagation).
        # The combination of new weights will be apply one each hidden node
        hidden_errors = np.dot(self.hidden_weights.T, output_errors)

        # Updates the weights for the links between the hidden and
        # output layers
        self.hidden_weights += self.learning_rate * np.dot((output_errors * final_outputs *
                                                            (1 - final_outputs)),
                                                           np.transpose(hidden_outputs))

        # Updates the weights for the links between the input and
        # hidden layers
        self.input_weights += self.learning_rate * np.dot((hidden_errors * hidden_outputs *
                                                           (1 - hidden_outputs)),
                                                          np.transpose(inputs))

    def query(self, inputs_list):
        """Returns the output to evaluate."""

        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.input_weights, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.hidden_weights, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
