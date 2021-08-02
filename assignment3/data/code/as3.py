import readImage as ri
import scipy.io
from numpy import array
import neuralNetwork as nn
import pickle


# The function normalize the pixels of images between 0 - 1
def normalization(data):
    normalization_dict = {}
    for i in range(len(data['x'])):
        minimum = min(data['x'][i])
        maximum = max(data['x'][i])
        im_list = []
        for j in range(len(data['x'][i])):
            pixel = data['x'][i][j]
            value = (pixel - minimum) / (maximum - minimum)
            im_list.append(value)
        normalization_dict[i] = array([im_list])
    return normalization_dict


#ri.read_data()
#train_data = scipy.io.loadmat('train.mat')
test_data = scipy.io.loadmat('test.mat')

#normalization_data = normalization(train_data)
test_normalize = normalization(test_data)
# If the count of hidden layel  is selected as 0 the code implements the single layel, otherwise multi hidden layels
#weights = nn.buildmultiNN(normalization_data, train_data['y'][0], 3, 50, 100, 16, 0.01, "sigmoid")
with open('model.p', 'rb') as fp:
    weights = pickle.load(fp)
nn.labeling(test_normalize, weights, test_data['y'][0])
