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

def plotDataSet():
	xArr, yArr = loadDataSet('ex0.txt')
	n = len(xArr)
	xcord = []; ycord = []
	for i in range(n):
		xcord.append(xArr[i][1]); ycord.append(yArr[i])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord, ycord, s = 20, c = 'g', alpha = 0.5)
	plt.title('DataSet')
	plt.xlabel('X')
	plt.show()


if __name__ == "__main__":
	plotDataSet()


