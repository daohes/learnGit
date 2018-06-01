# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []; labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == "lt":
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
# threshIneq表示两种比较模式(>threshVal为1，或<threshVal为1)，dimen是用来比较的特征的位置

def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump={}; bestClasEst = np.mat(np.zeros((m,1)))
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) /numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print("split: dim %d, thresh %.2f, thresh inequal: %s, the weightedError is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1)) / m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D:", D.T)
        alpha = float(.5*np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        #print("aggClassEst:", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        if errorRate == 0: break
    return weakClassArr, aggClassEst

def plotROC(predStrengths, classLabels):
	font = FontProperties(fname="c:/windows/fonts/simsun.ttc", size = 14)
	cur = (1.0, 1.0)
	ySum = 0.0
	numPosClas = np.sum(np.array(classLabels) == 1.0)
	yStep = 1 / float(numPosClas)
	xStep = 1 / float(len(classLabels) - numPosClas)

	sortedIndicies = predStrengths.argsort()

	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111) #一行一列第一张图
	for index in sortedIndicies.tolist()[0]:
		if classLabels[index] == 1.0:
			delX = 0; delY =yStep
		else:
			delX = xStep; delY = 0
			ySum += cur[1]
		ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'g')
		cur = (cur[0] - delX, cur[1] - delY)
	ax.plot([0,1],[0,1], 'b--') # 对角线
	plt.title('AdaBoost马疝气病检测系统ROC曲线', FontProperties = font)
	plt.xlabel('假阳性率', FontProperties = font)
	plt.ylabel('真阳性率', FontProperties = font)
	ax.axis([0, 1, 0, 1])
	# 设置x, y坐标轴最大最小值, v = [xmin, xmax, ymin, ymax]
	print('AUC面积为:', ySum * xStep)
	plt.show()
    
if __name__ == '__main__':
    dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr, 50)
    plotROC(aggClassEst.T, LabelArr)
