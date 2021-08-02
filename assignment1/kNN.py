import pandas as pd
import numpy as np
from scipy import spatial
import math


def optimizeData(trainSets, testSets, booksSet, usersSet):
    # this table is used to calculation of distance for kNN
    # this table is used for book information features
    testWithUser = pd.merge(testSets, usersSet, on='userID')
    trainWithUser = pd.merge(trainSets, usersSet, on='userID')
    # the code eliminate the some information
    boundaryUser = trainWithUser['userID'].value_counts()
    trainWithUser = trainWithUser[trainWithUser['userID'].isin(boundaryUser[boundaryUser >= 20].index)]
    boundaryBook = trainWithUser['rating'].value_counts()
    trainWithUser = trainWithUser[trainWithUser['rating'].isin(boundaryBook[boundaryBook >= 20].index)]
    # elimination of NaN  columns
    trainWithUser = trainWithUser.dropna()
    testWithUser = testWithUser.dropna()
    # check the year of publication
    booksSet.YearOfPublication = pd.to_numeric(booksSet.YearOfPublication, errors='coerce')
    booksSet.drop(booksSet[booksSet.YearOfPublication > 2018].index, inplace=True)
    # convert string to int for Age column and apply age period
    trainWithUser["Age"] = trainWithUser["Age"].astype("int")
    testWithUser["Age"] = testWithUser["Age"].astype("int")
    trainWithUser["userID"] = trainWithUser["userID"].astype("int")
    testWithUser["userID"] = testWithUser["userID"].astype("int")
    trainWithUser["rating"] = trainWithUser["rating"].astype("int")

    trainWithUser.drop(trainWithUser[(trainWithUser.Age > 90) & (trainWithUser.Age < 3)].index, inplace=True)
    testWithUser.drop(testWithUser[(testWithUser.Age > 90) & (testWithUser.Age < 3)].index, inplace=True)
    trainWithUser = trainWithUser.reset_index(drop=True)
    testWithUser = testWithUser.reset_index(drop=True)
    check = pd.merge(testWithUser, trainWithUser, on='userID')
    listUsers = check.userID.unique()

    if not check.empty:
        # the code creates the matrix to calculate similarity
        trainWithUser.drop(['Age', 'Location'], axis=1, inplace=True)
        kNN_matrix = trainWithUser.pivot_table(index='userID', columns='ISBN', values='rating')
        kNN_matrix.fillna(0, inplace=True)
        buildKNN(kNN_matrix, listUsers, testWithUser)
    else:
        print("test set elements are not enough to calculate similarity cause of dismatch")


def buildKNN(matrix, listUsers, testWithUser):
    listResults = []
    for i in range(len(listUsers)):
        listResult=[]
        v1 = matrix.loc[[listUsers[i]]].values
        # calculation of cosine Similarity

        for i in range(len(matrix.index)):

            v2 = matrix.loc[matrix.index[i]].values
            result = 1 - spatial.distance.cosine(v1, v2)
            if not math.isnan(result):
                if not 0 == result:
                    obj=[]
                    obj.append(result)
                    obj.append(matrix.index[i])
                    listResult.append(obj)

        listResults.append(listResult)
        # print(len(listResult))
    predictTheRating(listResults, testWithUser, matrix, listUsers,20)





def predictTheRating(listResults, testWithUser, matrix, listUsers,k):

    '''for j in range(len(listResult)):
        print(listResult[j])'''
    mae=0
    books = []
    ratings = []
    for i in range(len(listUsers)):
        book = testWithUser.loc[testWithUser['userID'] == listUsers[i]].ISBN.tolist()
        rating = testWithUser.loc[testWithUser['userID'] == listUsers[i]].rating.tolist()
        books.append(book)
        ratings.append(rating)
    for j in range(len(books)):
        for k in range(len(books[j])):
            total = 0
            denominator = 0
            total_weighted=0
            denominator_weighted=0
            for z in range(len(listResults)):
                listResults[z]=sorted(listResults[z], reverse=True)

                for w in range(min(k,len(listResults[z]))):

                    try:
                        total = total + matrix.loc[listResults[z][w][1], books[j][k]] * listResults[z][w][0]
                        denominator=denominator+listResults[z][w][0]
                        total_weighted = total + matrix.loc[listResults[z][w][1], books[j][k]] * 1/pow((1-listResults[z][w][0]),2)
                        denominator_weighted=denominator_weighted+1-listResults[z][w][0]

                    except KeyError :
                        continue

            try:
                # According to kNN

                print("kNN -> Book ISBN : "+books[j][k]+" 's rating is predicted : " +str(total/denominator)+" for user :"+str(listUsers[i]))

                print("kNN weighted -> Book ISBN : "+books[j][k]+" 's rating is predicted : " +str(total_weighted/denominator_weighted)+" for user :"+str(listUsers[i]))
                mae += total/denominator

            except ZeroDivisionError:
                continue

            except KeyError:
                continue
        meanAbsoluteError(mae,len(books[j]),sum(list(map(int, ratings[j]))))

def meanAbsoluteError(mae,length,totalRating):

    print(np.absolute(totalRating-mae)/length)