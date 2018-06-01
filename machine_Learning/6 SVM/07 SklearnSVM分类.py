# -*-coding:utf-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC

def img2vector(filename):
    """
    将32x32的二进制图像转换为1x1024向量。
    Parameters:
        filename - 文件名
    Returns:
        returnVect - 返回的二进制图像的1x1024向量
    """
    #创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    #打开文件
    fr = open(filename)
    #按行读取
    for i in range(32):
        #读一行数据
        lineStr = fr.readline()
        #每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    #返回转换后的1x1024向量
    return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = np.zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		hwLabels.append(classNumber)
		trainingMat[i,:] = img2vector('trainingDigits/%s' %(fileNameStr))
	clf = SVC(C=200,kernel='rbf')
	clf.fit(trainingMat,hwLabels)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' %(fileNameStr))
		classifierResult = clf.predict(vectorUnderTest)
		print("分类返回结果为:%d\t真实结果为:%d" %(classifierResult, classNumber))
		if(classifierResult != classNumber):
			errorCount += 1.0
	print("总共错了:%d个数据\t错误率为:%f%%" %(errorCount, errorCount/mTest * 100))

if __name__ == "__main__":
	handwritingClassTest()

    
