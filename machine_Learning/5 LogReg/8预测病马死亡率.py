#import matplotlib.pyplot as plt
import numpy as np
import random

"""
function: sigmoid 函数
modifytime: 20180327
"""
def sigmoid(inX):
    return (1.0 / (1 + np.exp(-inX)))

"""
function: 梯度上升算法
modifytime：同上
"""
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights += alpha * dataMatrix.transpose() * error
    return weights

"""
function: 改进的随机梯度上升算法
modifytime：同上
"""
def StocGradAscent1(dataMatIn, classLabels, numIter=150):
    dataMatrix = np.array(dataMatIn)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights += alpha * error * dataMatrix[randIndex]
            del(list(dataIndex)[randIndex])
    return weights#.getA()

"""
function: 预测
modifytime：同上
"""
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[:,0])) \
           != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100
    print("测试错误率为：%.2f%%" % errorRate)

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

if __name__ == '__main__':
    colicTest()

