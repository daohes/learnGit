# -*- coding: UTF-8 -*-

'''
function: 创建实验样本
modify time:
2018-03-21
'''
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], \
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],\
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],\
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],\
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],\
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

"""
function:
modify time:
2018-03-21
"""
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 取并集
    return list(vocabSet)

"""
function:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
modify time:
2018-03-21
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:   print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

if __name__ == "__main__":
    postingList, classVec = loadDataSet()
    print("postingList:\n", postingList)
    myVocabList = createVocabList(postingList)
    print("myVocabList:\n", myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print("trainMat:\n", trainMat)


















