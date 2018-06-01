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

def replaceNanWithMean():
	datMat = loadDataSet('secom.data',' ')
	numFeat = shape(datMat)[1]
	for i in range(numFeat):
		meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0], i])
		datMat[nonzero(isnan(datMat[:,i].A))[0], i] = meanVal
	return datMat

def plotPercentage(eigVals):
	x = range(len(eigVals)-1)
	#y = [100 * sum(eigVals[i])/sum(eigVals[:]) for i in range(0,len(eigVals))]
	y = [100 * sum(eigVals[:i])/sum(eigVals[:]) for i in range(1,len(eigVals))]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x, y, 'go-', alpha=.5)
	plt.xlim(0, 20)
	#plt.ylim(0, 65)
	plt.ylim(55, 100)
	plt.xticks(arange(0, 21, 5))
	#plt.yticks(arange(0, 65, 10))  
	plt.yticks(arange(55, 100, 10))
	plt.xlabel(u'主成分数目')
	#plt.ylabel(u'方差的百分比')
	plt.ylabel(u'方差的累积百分比')
	plt.show()


if __name__ == "__main__":
	dataMat = replaceNanWithMean()
	meanVals = mean(dataMat, axis=0)
	meanRemoved = dataMat - meanVals
	covMat = cov(meanRemoved, rowvar=0)
	eigVals, eigVects = linalg.eig(mat(covMat))
	plotPercentage(eigVals)