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

def plotlwlrRegression():
	xArr, yArr = loadDataSet('ex0.txt')
	yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)
	yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)
	yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)
	xMat = np.mat(xArr)
	yMat = np.mat(yArr)
	srtInd = xMat[:, 1].argsort(0) # argsort(0) 按列排序
	xSort = xMat[srtInd][:,0,0:2]
	'''
	[:,0,:] 表示												# [ [[1,2]], [[3,4]], [[5,6]] ]   [:,0]操作 ===> [ [1,2], [3,4], [5,6] ]   
	取全部行的第0个元素，步长为 :
	表示取0个元素中的全部元素，实际可以不传入第三个 :
	或者传入0:2，因为第0个元素中有两个元素
	'''
	fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10,8))
	axs[0].plot(xSort[:, 1], yHat_1[srtInd], c = 'r')
	axs[1].plot(xSort[:, 1], yHat_2[srtInd], c = 'r')
	axs[2].plot(xSort[:, 1], yHat_3[srtInd], c = 'r')
	axs[0].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'g', alpha = 0.5)
	axs[1].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'g', alpha = 0.5)
	axs[2].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'g', alpha = 0.5)

	axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', size = 14)
	axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', size = 14)
	axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003', size = 14)
	plt.setp(axs0_title_text, size=8, weight='bold', color='b')
	plt.setp(axs1_title_text, size=8, weight='bold', color='b')
	plt.setp(axs2_title_text, size=8, weight='bold', color='b')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()

def lwlr(testPoint, xArr, yArr, k = 1.0):
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	m = np.shape(xMat)[0]
	weights = np.mat(np.eye((m)))
	for j in range(m):
		diffMat = testPoint - xMat[j, :]
		weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))
	xTx = xMat.T * (weights * xMat)
	if np.linalg.det(xTx) == 0.0:
		print("矩阵为奇异矩阵，不能求逆")
		return
	ws = xTx.I * (xMat.T * (weights * yMat))
	return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
	m = np.shape(testArr)[0]
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i], xArr, yArr, k)
	return yHat

if __name__ == "__main__":
	plotlwlrRegression()
