import numpy as np

def binSplitDataSet(dataSet, feature, value):
	mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
	mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
	return mat0, mat1

if __name__ == "__main__":
	testMat = np.mat(np.eye(4))
	mat0, mat1 = binSplitDataSet(testMat, 1, .5)
	print("集合:\n", testMat)
	print("mat0:\n", mat0)
	print("mat1:\n", mat1)
	print(np.nonzero(testMat[:,1] > .5))