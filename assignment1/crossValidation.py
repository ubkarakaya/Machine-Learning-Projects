import numpy as np
import random
import pandas as pd
from random import randrange


def buildCrossValidation(ratingSets, foldNumber):

    #print(ratingSets)
    trainList=[]
    testList=[]

    for x in range(foldNumber):

        blockSize = round(len(ratingSets) / foldNumber)
        validationSize = round(len(ratingSets)/foldNumber)
        startIndex = x * blockSize
        finishIndex = x * blockSize + validationSize
        ratings_validation = ratingSets[startIndex:finishIndex]
        ratingSets = ratingSets.drop(ratingSets.index[startIndex:finishIndex])
        ratings_train = ratingSets
        trainList.append(ratings_train)
        testList.append(ratings_validation)
        #print(ratings_validation)
    return trainList,testList