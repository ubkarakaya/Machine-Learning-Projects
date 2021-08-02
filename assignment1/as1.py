import os
import sys
import readSplitFiles
import crossValidation
import kNN

ratingsSet, booksSet, usersSet = readSplitFiles.readFile()
# if we have testset and trainset we can pass this step
trainSets, testSets = crossValidation.buildCrossValidation(ratingsSet, 10)


for i in range(len(trainSets)):
    kNN.optimizeData(trainSets[i], testSets[i], booksSet, usersSet)
