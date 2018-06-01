from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
	fr = open(fileName)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	datArr = [list(map(float, line)) for line in stringArr]
	return mat(datArr)

def pca(dataMat, topNfeat=9999999):
	meanVals = mean(dataMat, axis=0)
	meanRemoved = dataMat - meanVals
	covMat = cov(meanRemoved, rowvar=0)
	eigVals, eigVects = linalg.eig(mat(covMat))
	eigValInd = argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVects = eigVects[:,eigValInd]
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat, reconMat

def plotPCA(dataMat, reconMat):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='x', s=60, c='g', alpha=.5)
	ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=60, c='r', alpha=.5)
	plt.show()

if __name__ == "__main__":
	dataMat = loadDataSet('testSet.txt')
	lowDMat, reconMat = pca(dataMat, 1)
	plotPCA(dataMat, reconMat)
