import matplotlib.pyplot as plt
import numpy as np

"""
function: 加载数据
modifytime:20180326
"""
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat

"""
function: sigmoid 函数
modifytime: 同上
"""
def sigmoid(inX):
    return float(1.0 / (1 + np.exp(-inX)))

"""
function: 随机梯度上升算法
modifytime：同上
"""
def StocGradAscent0(dataMatIn, classLabels):
    dataMatrix = np.array(dataMatIn)
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights += alpha * error * dataMatrix[i]
    return weights#.getA()

"""
function: 改进的随机梯度上升算法
modifytime：同上
"""
def StocGradAscent1(dataMatIn, classLabels, numIter=100):
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
function: 绘制边界
modifytime：20180326
"""
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'green', marker = 'o', alpha=.5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'red', marker = 'x', alpha=.5)
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] -  weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title(u'最佳决策边界')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
    
if __name__ == "__main__":
    dataMat, labelMat = loadDataSet()
    weights = StocGradAscent1(dataMat, labelMat)
    plotBestFit(weights)
