import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = list(map(float, curLine))                    #转化为float类型
		dataMat.append(fltLine)
	return dataMat

def binSplitDataSet(dataSet, feature, value):
	mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
	mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
	return mat0, mat1

def linearSolve(dataSet):
	m, n = np.shape(dataSet)
	X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))
	X[:,1:n] = dataSet[:, 0:n-1]; Y = dataSet[:,-1]
	xTx = X.T * X
	if np.linalg.det(xTx) == 0.0:
		print("奇异矩阵，不能求逆。\n")
		return
	ws = xTx.I * (X.T * Y)
	return ws, X, Y

def modelLeaf(dataSet):
	ws, X, Y = linearSolve(dataSet)
	return ws

def modelErr(dataSet):
	ws, X, Y = linearSolve(dataSet)
	yHat = X * ws
	return sum(np.power(Y- yHat, 2))

def chooseBestSplit(dataSet, leafType = modelLeaf, errType = modelErr, ops = (1,4)):
	'''函数说明:找到数据的最佳二元切分方式函数
        leafType - 生成叶结点
        regErr - 误差估计函数
        ops - 用户定义的参数构成的元组'''
	import types
	tolS = ops[0]; tolN = ops[1]
	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
		return None, leafType(dataSet)
	m, n = np.shape(dataSet)
	S = errType(dataSet)
	bestS = float('inf'); bestIndex = 0; bestValue = 0
	for featIndex in range(n - 1):
		for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
				continue
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS

	if (S - bestS) < tolS:
		return None, leafType(dataSet)
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
		return None, leafType(dataSet)
	return bestIndex, bestValue

def createTree(dataSet, leafType = modelLeaf, errType = modelErr, ops = (1, 10)):
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	if feat == None: return val

	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

def plotDataSet(filename, args):
	dataMat = loadDataSet(filename)                                        #加载数据集
	n = len(dataMat)                                                    #数据个数
	xcord = []; ycord = []                                                #样本点
	for i in range(n):                                                    
		xcord.append(dataMat[i][0]); ycord.append(dataMat[i][1])        #样本点
	fig = plt.figure()
	ax = fig.add_subplot(111)                                            #添加subplot
	ax.scatter(xcord, ycord, s = 20, c='b', alpha = .5)                #绘制样本点
	x = np.linspace(args.Min, args.Mid)
	plt.plot(x, args.k1*x+args.b1, color='r')
	x = np.linspace(args.Mid, args.Max)
	plt.plot(x, args.k2*x+args.b2, color='g')
	plt.title(u'数据集')                                                #绘制title
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()

class argsStruct:
	def __init__(self, treeMat):
		self.k1 = treeMat['right'].tolist()[1][0]
		self.b1 = treeMat['right'].tolist()[0][0]
		self.k2 = treeMat['left'].tolist()[1][0]
		self.b2 = treeMat['left'].tolist()[0][0]
		self.Mid = treeMat['spVal']
		self.Min = min(myMat[:,0]).tolist()[0][0]
		self.Max = max(myMat[:,0]).tolist()[0][0]	

if __name__ == '__main__':
	filename = 'exp2.txt'
	myDat = loadDataSet(filename)
	myMat = np.mat(myDat)
	treeMat = createTree(myMat)
	args = argsStruct(treeMat)
	plotDataSet(filename, args)


	