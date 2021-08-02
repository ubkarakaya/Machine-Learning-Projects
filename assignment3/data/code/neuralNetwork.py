import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from scipy.misc import logsumexp
import pickle

# activation functions are used between hidden layers
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def reLu(x):
    return np.maximum(x, 0)


# derivation of sigmoid value
def sigmoidPrime(s):
    return sigmoid(s) * (1 - sigmoid(s))


def SSE(y, prediction):
    return np.sum(np.square(y - prediction))


def softmax(y):
    return np.exp(y) / np.sum(y)


def activation(prediction, activation_p):
    if activation_p == "sigmoid":
        return sigmoid(prediction)

    elif activation_p == "tanh":
        return tanh(prediction)
    elif activation_p == "ReLU":
        return reLu(prediction)
    elif activation_p == "":
        return prediction


def normalize(x):
    maximum = max(x)
    minimum = min(x)
    for i in range(len(x)):
     x[i] = (x[i] - minimum) / (maximum - minimum)
    return x


def nLL(prediction, target, batchsize):

        ce = -np.sum(target * np.log(prediction))/batchsize
        return ce


class ML_Neural_Network(object):
    def __init__(self, input_size, hidden_layer_size, output_size, hidden_layer_count, learning_rate, activation):
        # initialize the parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_layer_size
        self.hiddenLayel = hidden_layer_count
        self.lastLayelSize = input_size
        self.lr = learning_rate
        self.dict_weights = {}  # the dictionary holds the weights
        self.dict_difference = {}  # the dictionary holds the errors
        self.layels = {}  # the dictionary holds the layels
        self.layel_count = 0  # layel count is used back propagation
        self.nll = 0
        self.keys = 0
        self.bias = 0.0
        self.activation = activation

    def generate_weight(self):
        for i in range(self.hiddenLayel):
            self.dict_weights[i] = np.random.rand(self.lastLayelSize, self.hiddenSize)
            self.lastLayelSize = self.dict_weights[i].shape[1]
            # Last Layel is output layel so its dimension is constant
        self.dict_weights[self.hiddenLayel] = np.random.randn(self.lastLayelSize, self.outputSize)

    def forward(self, x):
        # initialize the input layel as first layel
        self.layels[self.layel_count] = x

        # hidden layels and output layel
        for i in range(len(self.dict_weights.keys())):
            # all layels are balanced with using activation function
            self.layels[i + 1] = activation(np.dot(self.layels[i], self.dict_weights[i]), self.activation)
            self.layel_count += 1

        # return the prediction

        return softmax(self.layels[self.layel_count]) + self.bias

    def back_propagation(self, y, prediction):
        count = 0
        difference = np.subtract(y, prediction)
        delta = difference * sigmoidPrime(prediction)
        self.dict_difference[self.keys] = delta
        self.keys += 1

        for i in range(len(self.dict_weights.keys()) - 1, 0, -1):
            difference = self.dict_difference[count].dot(self.dict_weights[i].T)
            delta = difference * sigmoidPrime(self.layels[i])
            count += 1
            self.dict_difference[count] = delta

        #   Step for updating the weights
        #   input --> hidden weights
        #   hidden --> hidden
        #   ...         ...
        #   hidden --> output

        for i in range(len(self.dict_weights.keys())):
            self.dict_weights[i] += self.lr * self.layels[i].T.dot(self.dict_difference[count])
            count -= 1


def labeling(data, weights, labels):
    count = 0

    for key in data.keys():
        x = data[key]
        y = labels[key]
        layel = x
        for key2 in weights.keys():
            layel = sigmoid(np.dot(layel, weights[key2]))

        if np.argmax(layel) == y:
            count += 1
    print("Accuracy is :"+str(round(100 * (count / len(data.keys())))))


def buildmultiNN(data, labels, hidden_layer_count, hidden_layel_neuron, epoch, batchsize, learning_rate, activation):
    pre_list = []
    target_list = []
    loss_list = []
    label = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    label_table = array(label)
    NN = ML_Neural_Network(768, hidden_layel_neuron, 5, hidden_layer_count, learning_rate, activation)
    NN.generate_weight()
    limit = len(data.keys())
    for i in range(epoch):
        for start in range(0, limit - batchsize, batchsize):
            for key in range(start, start + batchsize):
                x = data[key]
                y = label_table[labels[key]]
                prediction = NN.forward(x)
                pre_list.append(prediction)
                target_list.append(y)
                NN.back_propagation(y, prediction)
                # before the next step temporary variables is renewed
                NN.dict_difference = {}
                NN.layels = {}
                NN.keys = 0
                NN.layel_count = 0
                NN.lastLayelSize = NN.inputSize

            loss_list.append(nLL(np.array(pre_list), np.array(target_list), batchsize))
            pre_list = []
            target_list = []
    '''plt.plot(normalize(loss_list))
    plt.ylabel('Loss Function')
    plt.show()'''
    with open('model.p', 'wb') as fp:
        pickle.dump(NN.dict_weights, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return NN.dict_weights
