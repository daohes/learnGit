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

#函数说明:使用局部加权线性回归计算回归系数w
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
	#函数说明:局部加权线性回归测试
	m = np.shape(testArr)[0]
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i], xArr, yArr, k)
	return yHat

def standRegres(xArr, yArr):
	#函数说明：无偏估计线性回归计算回归系数w
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	xTx = xMat.T * xMat
	if np.linalg.det(xTx) == 0.0:
		print("不能求逆")
		return
	ws = xTx.I * (xMat.T * yMat)
	return ws

def rssError(yArr, yHatArr):
	# 误差大小评价函数
	return ((yArr - yHatArr) **2).sum()

if __name__ == "__main__":
	abX, abY = loadDataSet('abalone.txt')
	print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
	yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
	yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
	yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
	print('k=0.1时,误差大小为:',rssError(abY[0:99], yHat01.T))
	print('k=1  时,误差大小为:',rssError(abY[0:99], yHat1.T))
	print('k=10 时,误差大小为:',rssError(abY[0:99], yHat10.T))
	print('')
	print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
	yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
	yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
	yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
	print('k=0.1时,误差大小为:',rssError(abY[100:199], yHat01.T))
	print('k=1  时,误差大小为:',rssError(abY[100:199], yHat1.T))
	print('k=10 时,误差大小为:',rssError(abY[100:199], yHat10.T))
	print('')
	print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
	print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
	ws = standRegres(abX[0:99], abY[0:99])
	yHat = np.mat(abX[100:199]) * ws
	print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))