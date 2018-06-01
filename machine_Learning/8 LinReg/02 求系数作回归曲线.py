import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1
	xArr = []; yArr = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		xArr.append(lineArr)
		yArr.append(float(curLine[-1]))
	return xArr, yArr

def standRegres(xArr, yArr):
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	xTx = xMat.T * xMat
	if np.linalg.det(xTx) == 0.0:
		print("矩阵为奇异矩阵，不能求逆")
		return
	ws = xTx.I * (xMat.T * yMat)
	return ws



def plotRegression():
	xArr, yArr = loadDataSet('ex0.txt')
	ws = standRegres(xArr, yArr)
	xMat = np.mat(xArr)
	yMat = np.mat(yArr)
	xCopy = xMat.copy()
	xCopy.sort(0) #对列排序
	yHat = xCopy * ws
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xCopy[:, 1], yHat, c = 'r')
	ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'g', alpha = .5)
	plt.title(u'数据集')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()

if __name__ == "__main__":
	plotRegression()