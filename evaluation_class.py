"""
If you want to run a single model, enter values for
input_nodes, hidden_nodes, output_nodes,
learning_rate and epochs in the instantiation of the object.
For a multyentry only enter the train and test datasetes.
"""

import matplotlib.pyplot as plt
import numpy as np
from ml_methods.mlp_number_detection import NumDetectionMLP
import time


class Evaluation:
    def __init__(self, dataset_1, dataset_2,
                 input_nodes=784, hidden_nodes=100,
                 output_nodes=10, learning_rate=0.1,
                 epochs=1, session=1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.session = session
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.model = NumDetectionMLP(self.input_nodes, self.hidden_nodes,
                                     self.output_nodes, self.learning_rate)

    def trainer(self):
        total_time = 0
        for epoch in range(1, self.epochs + 1):
            # Goes through all records in the training dataset
            start_time = time.time()
            for line in self.dataset_1:
                number_values = line.split(',')

                # Scales and shifts the inputs
                inputs = (np.asfarray(number_values[1:]) / 255 * 0.99) + 0.01

                # Assigning 0.01 to all the target output values
                # except the one with indext corresponding to
                # the number (label)
                targets = np.zeros(self.output_nodes) + 0.01

                # number_values[0] is the target label
                # for that specific line
                targets[int(number_values[0])] = 0.99
                self.model.train(inputs, targets)
            iteration_time = time.time() - start_time
            print('Epoch {} took {:3.2f} Seconds to run.'
                  .format(epoch, iteration_time))
            print('Training the data [{}{}] {}'.
                  format((epoch * 10 // self.epochs) * '.',
                         (10 - epoch * 10 // self.epochs) * ' ',
                         str(epoch * 100 // self.epochs) + '%'))
            total_time += iteration_time
        return total_time

    def evaluation(self):
        """Generates a vector for computing the performance."""
        training_result = []

        # Checking each line in test dataset
        for line in self.dataset_2:
            numbers_value = line.split(',')
            number = int(numbers_value[0])

            # Scales and shifts the inputs
            inputs = (np.asfarray(numbers_value[1:]) / 255 * 0.99) + 0.01
            outputs = self.model.query(inputs)

            # Greates value's index among the number values
            label_index = np.argmax(outputs)

            if label_index == number:
                training_result.append(1)
            else:
                training_result.append(0)

        # Evaluating the performance
        scoreboard_array = np.asarray(training_result)

        return scoreboard_array.sum() / scoreboard_array.size

    def performance(self):
        training_time = self.trainer()
        perf = self.evaluation()
        return '{0:<11}{1:<11}{2:<12}{3:<12}{4:<8}{5:<10}{6:<15}{7:<3.2f}\n' \
            .format(self.session, self.input_nodes,
                    self.hidden_nodes, self.output_nodes,
                    self.learning_rate, self.epochs,
                    perf, training_time)

    def multievaluation(self, evaluation_num):
        """In order to compare different structures
        use can input different values for the data members"""
        performances = ''
        for session in range(1, evaluation_num + 1):
            print('Session {}'.format(session))
            self.epochs = int(input('Number of epochs for the current session: '))
            self.input_nodes = int(input('Number of input nodes: '))
            self.hidden_nodes = int(input('Number of hidden nodes: '))
            self.output_nodes = int(input('Number of output nodes: '))
            self.learning_rate = float(input('Learning rate: '))
            performances += self.performance()
        return performances

    @staticmethod
    def visualize(dataset, line):
        """Plots any line from any number detection dataset."""
        all_values = dataset[line].split(',')
        image_array = np.asfarray(all_values[1::]).reshape((28, 28))
        print('The number to be detected:', all_values[0])
        print('The corresponding matrix of its image: ')
        print(image_array)
        plt.imshow(image_array, cmap='Greys',
                   interpolation='None')
        plt.savefig('number.png')
        # plt.show()

    def __str__(self):
        evaluation_num = int(input('How many models do you want to apply? '))
        result = ''
        if evaluation_num == 1:
            result = self.performance()
        elif evaluation_num == 0:
            print('You have not applied any model.')
        else:
            result = self.multievaluation(evaluation_num)
        return 'Session    InNodes    HidNodes    OutNodes    Rate    ' \
               'Epochs    Performance    Total time\n' + result +\
               '\n' + \
               'InNodes: Number of Input Nodes\n' \
               'OutNodes: Number of Output Nodes\n' \
               'HidNodes: Number of Output Nodes\n' \
               'Rate: Learning Rate\n' \
               'Multiperceptron model with one hidden layer\n' \
               'Feedforward Artificial Neural Network\n' \
               'Backpropagation\n' \
               'Activation Function: Sigmoid\n'


def data(dataset):
    data_file = open(dataset, 'r')
    data_list = data_file.readlines()
    data_file.close()
    return data_list


if __name__ == "__main__":
    # A 0.97 performance can be expected applying 784 input nodes,
    # 200 hidden nodes, 10 output nodes and learning rate of 0.2
    # through seven epochs.
    train_list = data('datasets/mnist_train.csv')
    test_list = data('datasets/mnist_test.csv')
    eval_01 = Evaluation(train_list, test_list)
    eval_01.visualize(train_list, 20)
    # print(eval_01)







