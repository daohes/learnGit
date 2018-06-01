# -*- coding: UTF-8 -*-
import numpy as np

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
        vocabSet = vocabSet | set(document)
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

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 任选一个文件，这个文件是侮辱文件的概率
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 侮辱文件类别下，各个单词出现的次数
            p1Denom += sum(trainMatrix[i]) # 侮辱文件类别下，出现单词的总个数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom # 侮辱文件下各个单词出现的概率
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive

if __name__ == "__main__":
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    print("postingList:\n", postingList)
    print("myVocabList:\n", myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print("trainMat:\n",trainMat)
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    print("p0V(非侮辱文件下各个单词出现的概率):\n", p0V)
    print("p1V(侮辱文件下各个单词出现的概率):\n", p1V)
    print("classVec:\n", classVec)
    print("pAb(任选一个文件，这个文件是侮辱文件的概率):\n", pAb)






