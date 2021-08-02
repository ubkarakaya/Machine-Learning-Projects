from nltk import ngrams
import collections
import math
from sklearn.feature_extraction import text


def findCount(word, dict):
    for key in dict.keys():
        if word == key:
            return dict[key][0]
        else:
            return 0


def findKeyWords(list):
    words = []
    for i in range(len(list)):

        array = list[i].split()

        for j in range(len(array)):
            words.append(array[j])
    frequencyOfWords = collections.Counter()
    frequencyOfWords.update(words)
    return frequencyOfWords


def addUnseenWords(test,real,fake):

    for key in test.keys():
        if not key in real.keys():
            real[key] = 1
        if not key in fake.keys():
            fake[key] = 1
    return real, fake

def buildUnigram(frequency, ngramDict):
    totalCount = sum(frequency.values());
    for key, value in frequency.items():
        # Laplace Smoothing #
        ngramDict[key] = [value, math.log((value +1) / (totalCount + len(frequency.keys())))]
    return ngramDict


def buildBigram(list, grams, bigramDict):
    bigram = []
    for i in range(len(list)):
        array = list[i].split()
        bigram.append(ngrams(array, 2))
    for i in range(len(bigram)):
        for gram in bigram[i]:
            grams.append(gram)
    frequencyOfBiGram = collections.Counter()
    frequencyOfBiGram.update(grams)
    for key, value in frequencyOfBiGram.items():
        bigramDict[key] = value
    return bigramDict


def calculateProbablity(testNgram, realNgram, fakeNgram, probReal, probFake):
    for grams in testNgram:
        grams = ''.join(grams)
        if grams in realNgram.keys():
            probReal = probReal + realNgram[grams][1]

        if grams in fakeNgram.keys():
            probFake = probFake + fakeNgram[grams][1]

    if probReal == max(probFake, probReal):
        return "real"
    else:
        return "fake"


def newsLabelUni(testFrame, unigramDictReal, unigramDictFake, probReal, probFake):
    right_prediction = 0
    for i in range(len(testFrame.Id)):
        testUnigram = ngrams(testFrame.Id[i].split(), 1)

        if calculateProbablity(testUnigram, unigramDictReal, unigramDictFake, probReal,
                               probFake) == \
                testFrame.Category[i]:
            right_prediction = right_prediction + 1
    print("Accuracy of the Unigram Model : " + str(round(100 * (right_prediction / len(testFrame.Id)))) + "%")


def calculateProbBigram(testNgram, frequencyReal, frequencyFake, uniReal, uniFake, realNgram, fakeNgram, probReal,
                        probFake):
    deliminator = len(frequencyReal)
    deliminatorFake = len(frequencyFake)

    for grams in testNgram:

        if grams in realNgram.keys():
            probReal = probReal + math.log((realNgram[grams] + 1) / (findCount(grams[0], uniReal) + deliminator))
        else:

                probReal = probReal + math.log(1 / (findCount(grams[0], uniReal) + deliminator))

        if grams in fakeNgram.keys():
            probFake = probFake + math.log((fakeNgram[grams] + 1) / (findCount(grams[0], uniFake) + deliminatorFake))
        else:
                probFake = probFake + math.log(1 / (deliminatorFake + findCount(grams[0], uniFake)))

    if probReal == max(probFake, probReal):
        return "real"
    else:
        return "fake"


def newsLabelBi(testFrame, frequencyReal, frequencyFake, unigramDictReal, unigramDictFake, bigramDictReal,
                bigramDictFake, probReal, probFake):
    right_prediction = 0
    realWords = []
    fakeWords = []

    for i in range(len(testFrame.Id)):
        sentence = "<s> " + testFrame.Id[i] + " <n>"
        words = sentence.split()
        testBigram = ngrams(sentence.split(), 2)
        label = calculateProbBigram(testBigram, frequencyReal, frequencyFake, unigramDictReal, unigramDictFake,
                                    bigramDictReal, bigramDictFake, probReal, probFake)

        if label == testFrame.Category[i]:
            right_prediction = right_prediction + 1

        if label == "real":
            for i in range(len(words)):
                realWords.append(words[i])
        else:
            for i in range(len(words)):
                fakeWords.append(words[i])
    print("Accuracy of the Bigram Model  : " + str(round(100 * (right_prediction / len(testFrame.Id)))) + "%")
    listWords(realWords, fakeWords, unigramDictReal, unigramDictFake)


def listWords(realWords, fakeWords, dictReal, dictFake):
    absentReal = {}
    absentFake = {}

    for key,value in dictReal.items():
        if not key in realWords:
            absentReal[key] = value[0]

    for key,value in dictFake.items():
        if not key in fakeWords:
            absentFake[key] = value[0]

    frequencyOfReals = collections.Counter()
    frequencyOfReals.update(realWords)

    frequencyOfFakes = collections.Counter()
    frequencyOfFakes.update(fakeWords)
    # The code sorts the dictionary according to its value #
    sortedAbsentReal = [(k, absentReal[k]) for k in sorted(absentReal, key=absentReal.get, reverse=True)]
    sortedAbsentFake = [(k, absentFake[k]) for k in sorted(absentFake, key=absentFake.get, reverse=True)]

    print("List the 10 words whose presence most strongly predicts that the news is real :")
    print(frequencyOfReals.most_common(10))
    print("List the 10 words whose presence most strongly predicts that the news is fake :")
    print(frequencyOfFakes.most_common(10))
    print("List the 10 words whose absence most strongly predicts that the news is real :")
    print(sortedAbsentReal[:10])
    print("List the 10 words whose absence most strongly predicts that the news is fake :")
    print(sortedAbsentFake[:10])

    # Effects of Stop Words #
    add = ["<s>", "<n>"]
    stop_words = text.ENGLISH_STOP_WORDS.union(add)
    filtered_realNews = {}
    filtered_fakeNews = {}

    for key, value in frequencyOfReals.items():
        if key not in stop_words:
            filtered_realNews[key] = value

    for key, value in frequencyOfFakes.items():
        if key not in stop_words:
            filtered_fakeNews[key] = value

    filteredReals = collections.Counter()
    filteredReals.update(filtered_realNews)

    filteredFakes = collections.Counter()
    filteredFakes.update(filtered_fakeNews)
    print("List the 10 non-stopwords that most strongly predict that the news is real :")
    print(filteredReals.most_common(10))
    print("List the 10 non-stopwords that most strongly predict that the news is fake :")
    print(filteredFakes.most_common(10))