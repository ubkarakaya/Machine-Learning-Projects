
import readFiles as rd
import Ngrams as ng
import math

# initialization of dictionaries to hold the N-Gram Models (Unigram and Bigram)
unigramDictReal = {}
unigramDictFake = {}
unigramDictTest ={}
bigramDictReal = {}
bigramDictFake = {}
gramsReal = []
gramsFake = []

list_real, list_fake,list_test, testFrame = rd.readFiles()


# add a specific expression to detect head of the sentence  #
for i in range(len(list_real)):
    list_real[i] = list_real[i][0:-2]
    list_real[i] = "<s> " + list_real[i]+" <n>"
for j in range(len(list_fake)):
    list_fake[j] = list_fake[j][0:-2]
    list_fake[j] = "<s> "+ list_fake[j]+" <n>"

# Most commonly keywords #
print("Most commonly 3 keywords for Real News:")
frequencyReal = ng.findKeyWords(list_real)
print(frequencyReal.most_common(5)[2:5])
print("Most commonly 3 keywords for Fake News:")
frequencyFake = ng.findKeyWords(list_fake)
print(frequencyFake.most_common(5)[2:5])

frequencyTest = ng.findKeyWords(list_test)
frequencyReal, frequencyFake = ng.addUnseenWords(frequencyTest, frequencyReal, frequencyFake)
unigramDictReal = ng.buildUnigram(frequencyReal, unigramDictReal)
unigramDictFake = ng.buildUnigram(frequencyFake, unigramDictFake)

bigramDictReal = ng.buildBigram(list_real, gramsReal, bigramDictReal)
bigramDictFake = ng.buildBigram(list_fake, gramsFake, bigramDictFake)

probReal = len(list_real) / (len(list_real)+len(list_fake))
probFake = len(list_fake) / (len(list_real)+len(list_fake))

ng.newsLabelUni(testFrame, unigramDictReal, unigramDictFake
                , math.log(probReal), math.log(probFake))

ng.newsLabelBi(testFrame, frequencyReal, frequencyFake,unigramDictReal,unigramDictFake, bigramDictReal, bigramDictFake
                , math.log(probReal), math.log(probFake))