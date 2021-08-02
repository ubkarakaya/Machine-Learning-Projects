import csv
import os
import sys
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt / 10)
        decrement = True

# The function takes an argument and reads the all csv files
def readFile():
    file = []
    for filename in os.listdir(sys.argv[1]):
        file.append(sys.argv[1] + "/" + filename)
    # Book Ratings#
    dataRatings = pd.read_csv(file[0], sep=';', header=None, engine='python')
    dataRatings.columns = ['userID','ISBN','rating']
    # Books#
    dataBooks = pd.read_csv(file[1], sep=';', header=None, engine='python', error_bad_lines=False,warn_bad_lines=False)
    dataBooks.columns = ['ISBN', 'Book-Title', 'Book-Author', 'YearOfPublication', 'Publisher','Url-S', 'Url-M', 'Url-L']
    # deleting last 3 columns cause of unnecessary for operation #
    dataBooks.drop(['Url-S', 'Url-M', 'Url-L'], axis=1, inplace=True)
    # Users#
    dataUsers = pd.read_csv(file[2], sep=';', header=None, engine='python', error_bad_lines=False,warn_bad_lines=False)
    dataUsers = dataUsers[1:]
    #dataUsers[0] = [e[1:] for e in dataUsers[0]]
    dataUsers.columns = ['userID','Location','Age']
    # The code picks the users from Canada and USA
    dataUsers = dataUsers[dataUsers.Location.str.contains('usa' or 'canada', na=False)]
    return dataRatings, dataBooks, dataUsers


