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

def regularize(xMat, yMat):
	#数据标准化
	inxMat = xMat.copy()
	inyMat = yMat.copy()
	yMean = np.mean(yMat, axis = 0)
	inyMat = yMat - yMean
	inMeans = np.mean(inxMat, axis = 0)
	inVar = np.var(inxMat, axis = 0)
	inxMat = (inxMat - inMeans) / inVar
	return inxMat, inyMat

def rssError(yArr, yHatArr):
	return ((yArr-yHatArr)**2).sum()

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	xMat, yMat = regularize(xMat, yMat)
	m, n = np.shape(xMat)
	returnMat = np.zeros((numIt, n))
	ws = np.zeros((n, 1))
	wsTest = ws.copy()
	wsMax = ws.copy()
	for i in range(numIt):
		lowestError = float('inf')
		for j in range(n):
			for sign in [-1, 1]:
				wsTest = ws.copy()
				wsTest[j] += eps * sign
				yTest = xMat * wsTest
				rssE = rssError(yMat.A, yTest.A)
				if rssE < lowestError:
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i,:] = ws.T
	return returnMat

def plotstageWiseMat():
	xArr, yArr = loadDataSet('abalone.txt')
	returnMat = stageWise(xArr, yArr, 0.005, 1000)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(returnMat)
	ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系')
	ax_xlabel_text = ax.set_xlabel(u'迭代次数')
	ax_ylabel_text = ax.set_ylabel(u'回归系数')
	plt.setp(ax_title_text, size = 15, weight = 'bold', color = 'red')
	plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
	plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
	plt.show()
if __name__ == '__main__':
	plotstageWiseMat()