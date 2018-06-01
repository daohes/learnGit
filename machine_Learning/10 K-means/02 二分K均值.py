#coding=utf-8
from numpy import *
import matplotlib.pyplot as plt
#load data
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines(): #for each line
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #这里和书中不同 和上一章一样修改
        dataMat.append(fltLine)
    return dataMat

#distance func
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB) 向量AB的欧式距离

#init K points randomly
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

#K-均值算法:
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    #参数：dataset,num of cluster,distance func,initCen
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))#store the result matrix,2 cols for index and error
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):#for every points
            minDist = inf;minIndex = -1#init
            for j in range(k):#for every k centers，find the nearest center
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:#if distance is shorter than minDist
                    minDist=distJI;minIndex=j# update distance and index(类别)
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                #此处判断数据点所属类别与之前是否相同（是否变化，只要有一个点变化就重设为True，再次迭代）
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        # update k center
        for cent in range(k):
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment

#K-均值算法:
def biKmeans(dataSet, k, distMeas = distEclud):
	m = shape(dataSet)[0]	# 获取数据集样本数
	clusterAssment = mat(zeros((m,2)))	#存放数据所属类别及与质心的误差值
	centroid0 = mean(dataSet, axis=0).tolist()[0]	#计算数据集质心
	centList = [centroid0]
	for j in range(m):
		clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:]) ** 2 #计算各数据与质心的误差并将其标注为类别0 1 2 3...
	while (len(centList) < k):
		lowestSSE = float('inf')
		for i in range(len(centList)):
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]	#数据集中属于同一质心类的数据筛选出来
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)		#kMeans方法计算质心和误差
			sseSplit = sum(splitClustAss[:,1])	#计算属于该簇误差之和
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])	#计算不属于该簇的误差
			print("sseSplit, and not sseSplit: ", sseSplit, sseNotSplit)
			if (sseSplit + sseNotSplit) < lowestSSE:	#若总误差最小，就用质心i划分，并计算误差和相应的分割簇
				bestCentToSplit = i
				bestNewCents = centroidMat
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)	#不属于该簇的标记加1
		bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit	#属于该簇的标记为该簇
		print('the bestCentToSplit is: ', bestCentToSplit)
		print('the len of bestClustAss is: ', len(bestClustAss))
		centList[bestCentToSplit] = bestNewCents[0,:]	#第一个簇
		centList.append(bestNewCents[1,:])				#第二个簇
		clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss	
	return centList, clusterAssment

def plotClustResult(dataSet, centList, clusterAssment):
	centlist = []
	clust = [myNewAssments[:,0].tolist()[i][0]+1 for i in range(len(myNewAssments))]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataSet[:,0], dataSet[:,1], 15.0*array(clust), 15.0*array(clust))
	plt.show()


if __name__ == "__main__":
	dataSet = loadDataSet('testSet2.txt')
	datMat = mat(dataSet)
	centList, myNewAssments = biKmeans(datMat, 3)
	print([centList[i].A for i in range(len(centList))])
	plotClustResult(datMat.A, centList, myNewAssments)

