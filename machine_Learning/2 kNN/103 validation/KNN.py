from numpy import *
import operator

"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果

Modify:
    2018-03-11
"""

def classify0(inX, dataSet, labels, k):
     dataSetSize = dataSet.shape[0] #numpy函数shape[0]返回dataSet的行数
     diffMat = tile(inX, (dataSetSize,1)) - dataSet  #在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
     sqDiffMat = diffMat ** 2
     sqDistances = sqDiffMat.sum(axis = 1)
     distances = sqDistances ** .5
     sortedDistIndicies = distances.argsort()
     classCount = {}
     for i in range(k):
          voteIlabel = labels[sortedDistIndicies[i]]
          classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
     sortedClassCount = sorted(classCount.items(),\
                               key=operator.itemgetter(1), reverse = True)
     return sortedClassCount[0][0]

"""
函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量

Modify:
    2018-03-11
"""
def file2matrix(filename):
     fr = open(filename)
     arrayOfLines = fr.readlines()
     numberOfLines = len(arrayOfLines)
     returnMat = zeros((numberOfLines,3))
     classLabelVector = []
     index = 0
     for line in arrayOfLines:
          line = line.strip() # 去除两端空格,中间间隔仍保留
          listFromLine = line.split('\t')
          returnMat[index,:] = listFromLine[0:3] #将数据前三列提取出来,存放到returnMat矩阵中

          if listFromLine[-1] == 'didntLike':
               classLabelVector.append(1)
          elif listFromLine[-1] == 'smallDoses':
               classLabelVector.append(2)
          elif listFromLine[-1] == 'largeDoses':
               classLabelVector.append(3)
          index += 1
     return returnMat, classLabelVector

"""
函数说明:对数据进行归一化

Parameters:
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值

Modify:
    2018-03-11
"""

def autoNorm(dataSet):
     minVals = dataSet.min(0) # min(0)  每列的最小值    min(1) 每行最小值
     maxVals = dataSet.max(0)
     ranges = maxVals - minVals
     
     normDataSet = zeros(shape(dataSet)) #shape(dataSet)返回dataSet维度
     m = dataSet.shape[0]

     normDataSet = dataSet - tile(minVals, (m, 1))
     normDataSet /= tile(ranges, (m, 1))

     return normDataSet, ranges, minVals


def datingClassTest():
     filename = 'datingTestSet.txt'
     datingDataMat, datingLabels = file2matrix(filename)
     hoRatio = 0.10

     normMat, ranges, minVals = autoNorm(datingDataMat)
     m = normMat.shape[0]
     numTestVecs = int(m * hoRatio)
     errorCount = 0.0

     for i in range(numTestVecs):
          classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
                                       datingLabels[numTestVecs:m], 4)
          print("分类结果:%d\t真实结果:%d" % (classifierResult, datingLabels[i]))
          if classifierResult != datingLabels[i]:
               errorCount += 1.0
     print('错误率:%f%%' %(errorCount/float(numTestVecs)*100))

if __name__ == '__main__':
     datingClassTest()
















     

