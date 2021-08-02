
import pandas as pd

def readFiles():

    list_real = []
    list_fake = []
    list_test = []
    with open('data/clean_fake-Train.txt', 'r+') as f:
        for line in f.readlines():
            list_fake.append(line)
    with open('data/clean_real-Train.txt', 'r+') as f:
        for line in f.readlines():
            list_real.append(line)

    testFrame = pd.read_csv('data/test.csv')
    testFrame.columns = ['Id', 'Category']
    list_test = testFrame['Id'].tolist()
    return list_real, list_fake,list_test, testFrame
