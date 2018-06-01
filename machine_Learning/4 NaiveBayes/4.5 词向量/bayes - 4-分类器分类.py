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

"""
function:朴素贝叶斯分类器训练函数
Modify:20180321
"""
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

"""
函数说明:朴素贝叶斯分类器分类函数
Modify:20180321
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    #print('p0:', p0)
    #print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0

"""
function:测试
modify：20180321
"""
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, "属于侮辱类")
    else:
        print(testEntry, "属于非侮辱类")
    testEntry = ['stupid','garbage','worthless']

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, "属于侮辱类")
    else:
        print(testEntry, "属于非侮辱类")

if __name__ == "__main__":
    testingNB()













