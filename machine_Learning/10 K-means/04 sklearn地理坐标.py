from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#distance func
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB) 向量AB的欧式距离

# 返回地球表面两点之间的距离 单位英里 输入经纬度(度) 球面余弦定理
def distSLC(vecA, vecB=[0.,0.]):
	a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
	b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi* (vecB[0,0]-vecA[0,0]) / 180)
	return arccos(a + b)*6371.0

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

def clusterClubs(numClust=5):
	datList = []
	for line in open('places.txt').readlines():
		lineArr = line.split('\t')
		datList.append([float(lineArr[4]), float(lineArr[3])])
	datMat = mat(datList)
	for i in range(len(datMat)):
		print(i)
	myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas = distSLC)
	fig = plt.figure()
	rect = [.1, .1, .8, .8]
	scatterMarkers = ['s','o','^','8','p','d','v','h','>','<']
	axprops = dict(xticks=[], yticks=[])
	ax0 = fig.add_axes(rect, label='ax0', **axprops)
	imgP = plt.imread('Portland.png')
	ax0.imshow(imgP)
	ax1 = fig.add_axes(rect, label='ax1', frameon=False)
	for i in range(numClust):
		ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A == i)[0], :]
		markerStyle = scatterMarkers[i % len(scatterMarkers)]
		ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
		ax1.scatter(myCentroids[i].tolist()[0][0], myCentroids[i].tolist()[0][1], marker='+', s=300)
	plt.show()

if __name__ == "__main__":

	clusterClubs(5)