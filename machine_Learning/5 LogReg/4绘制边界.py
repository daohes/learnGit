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
    return 1.0 / (1 + np.exp(-inX))

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
    # scatter中x横坐标，y纵坐标，s大小，c颜色，marker样式，alpha透明度
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] -  weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title(u'最佳决策边界')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
    
if __name__ == "__main__":
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)
