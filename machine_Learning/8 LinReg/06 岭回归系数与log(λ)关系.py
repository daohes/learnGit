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

def ridgeRegres(xMat, yMat, lam = .2):
	#函数说明：ridge regression
	xTx = xMat.T * xMat
	denom = xTx + np.eye(np.shape(xMat)[1]) * lam
	if np.linalg.det(denom) == 0.0:
		print("矩阵为奇异矩阵，不能求逆")
		return
	ws = denom.I * (xMat.T * yMat)
	return ws

def ridgeTest(xArr, yArr):
	xMat = np.mat(xArr); yMat = np.mat(yArr).T

	yMean = np.mean(yMat, axis = 0)
	yMat = yMat - yMean
	xMeans = np.mean(xMat, axis = 0)
	xVar = np.var(xMat, axis=0)
	xMat = (xMat - xMeans) / xVar
	numTestPts = 30
	wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
	for i in range(numTestPts):
		ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
		wMat[i, :] = ws.T
	return wMat

def plotwMat():
	abX, abY = loadDataSet('abalone.txt')
	ridgeWeights = ridgeTest(abX, abY)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ridgeWeights)
	ax_title_text = ax.set_title(u'log(λ)与回归系数的关系', size = 14)
	ax_xlabel_text = ax.set_xlabel(u'log(λ)')
	ax_ylabel_text = ax.set_ylabel(u'回归系数')
	plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
	plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
	plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
	plt.show()
if __name__ == '__main__':
	plotwMat()