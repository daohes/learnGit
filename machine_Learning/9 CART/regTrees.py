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

def regLeaf(dataSet):
	#生成叶节点
	return np.mean(dataSet[:,-1])

def regErr(dataSet):
	#误差估计函数
	return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

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

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
	'''函数说明:找到数据的最佳二元切分方式函数
        leafType - 生成叶结点
        regErr - 误差估计函数
        ops - 用户定义的参数构成的元组'''
	import types
	'''tolS允许的误差下降值,tolN切分的最少样本数'''
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

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	if feat == None: return val

	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

def isTree(obj):
	import types
	return (type(obj).__name__ == "dict")

def regTreeEval(model, inDat):
	return float(model)

def modelTreeEval(model, inDat):
	n = np.shape(inDat)[1]
	X = np.mat(np.ones((1, n+1)))
	X[:,1:n+1] = inDat
	return float(X*model)

def treeForeCast(tree, inData, modelEval = regTreeEval):
	if not isTree(tree): return modelEval(tree, inData)
	if inData[tree['spInd']] > tree['spVal']:
		if isTree(tree['left']):
			return treeForeCast(tree['left'], inData, modelEval)
		else:
			return modelEval(tree['left'], inData)
	else:
		if isTree(tree['right']):
			return treeForeCast(tree['right'], inData, modelEval)
		else:
			return modelEval(tree['right'], inData)
			
def createForeCast(tree, testData, modelEval=regTreeEval):
	m = len(testData)
	yHat = np.mat(np.zeros((m,1)))
	for i in range(m):
		yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
	return yHat

if __name__ == '__main__':
	trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
	testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
	regTree = createTree(trainMat, ops=(1,20))
	regYHat = createForeCast(regTree, testMat[:,0])
	modelTree = createTree(trainMat, modelLeaf, modelErr, ops=(1,20))
	modelYHat = createForeCast(modelTree, testMat[:,0], modelTreeEval)
	ws, X, Y = linearSolve(trainMat)
	linearYHat=np.mat(np.zeros((np.shape(modelYHat))))
	for i in range(np.shape(testMat)[0]):
		linearYHat[i] = testMat[i,0] * ws[1,0] + ws[0,0] 
	print('回归树R^2=',np.corrcoef(regYHat, testMat[:,1], rowvar=0)[0,1])
	print('模型树R^2=',np.corrcoef(modelYHat, testMat[:,1], rowvar=0)[0,1])
	print('线性回归R^2=',np.corrcoef(linearYHat, testMat[:,1], rowvar=0)[0,1])
