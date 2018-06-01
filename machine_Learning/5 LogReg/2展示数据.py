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
function: 绘制数据集
modifytime：20180326
"""
def plotDataSet():
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
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 'o', alpha=.5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', marker = 'x', alpha=.5)
    plt.title(u'数据集')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    plotDataSet()
