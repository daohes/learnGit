# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():
    """
    创建单层决策树的数据集
    Parameters:
        无
    Returns:
        dataMat - 数据矩阵
        classLabels - 数据标签
    """
    datMat = np.matrix([[ 1. ,  2.1],
        [ 1.5,  1.6],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == "lt":
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
# threshIneq表示两种比较模式(>threshVal为1，或<threshVal为1)，dimen是用来比较的特征的位置

def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump={}; bestClasEst = np.mat(np.zeros((m,1)))
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) /numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh inequal: %s, the weightedError is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def showStumpClassify(dataMat, labelMat, thresh):
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(dataMat.shape[0]):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)                                             #转换为numpy矩阵
    data_minus_np = np.array(data_minus)                                         #转换为numpy矩阵
    plt.scatter((data_plus_np.T)[0], (data_plus_np.T)[1])        #正样本散点图
    plt.scatter((data_minus_np.T)[0], (data_minus_np.T)[1])     #负样本散点图
    plt.axhline(thresh,c='g')   #做出分界线
    plt.show()

if __name__ == '__main__':
    dataArr,classLabels = loadSimpData()
    D = np.mat(np.ones((5,1))/5)
    bestStump, minError, bestClasEst = buildStump(dataArr, classLabels, D)
    print('bestStump:\n', bestStump)
    print('minError:\n', minError)
    print('bestClasEst:\n', bestClasEst)
    showStumpClassify(dataArr,classLabels,bestStump['thresh'])