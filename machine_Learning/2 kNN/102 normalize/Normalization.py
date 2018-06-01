from numpy import *

'''函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量

Modify:
    2018-03-11
'''

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
          returnMat[index,:] = listFromLine[0:3]

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
     m = dataSet.shape[0] #返回dataSet行数

     normDataSet = dataSet - tile(minVals, (m, 1)) #重复m行1列 次的minVals矩阵
     normDataSet /= tile(ranges, (m, 1))

     return normDataSet, ranges, minVals

if __name__ == '__main__':
     filename = "datingTestSet.txt"

     datingDataMat, datingLabels = file2matrix(filename)
     normDataSet, ranges, minVals = autoNorm(datingDataMat)
     print(normDataSet)
     print(ranges)
     print(minVals)

















     

