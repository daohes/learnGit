import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():                                     #逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])      #添加数据
        labelMat.append(float(lineArr[2]))                          #添加标签
    return dataMat,labelMat

def showDataSet(dataMat, labelMat):
	data_plus = []
	data_minus = []
	for i in range(len(dataMat)):
		if labelMat[i] > 0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np = np.array(data_plus)
	data_minus_np = np.array(data_minus)
	plt.scatter((data_plus_np.T)[0], (data_plus_np.T)[1])
	plt.scatter((data_minus_np.T)[0], (data_minus_np.T)[1])
	plt.show()

if __name__ == "__main__":
	dataArr, labelArr = loadDataSet('testSetRBF.txt')
	showDataSet(dataArr, labelArr)

    
