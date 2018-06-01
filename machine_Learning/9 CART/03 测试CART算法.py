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

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
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

if __name__ == '__main__':
	myDat = loadDataSet('ex2.txt')
	myMat = np.mat(myDat)
	feat, val = chooseBestSplit(myMat, regLeaf, regErr, (1,4))
	print(feat)
	print(val)