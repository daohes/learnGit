#！\usr\bin\env python
from numpy import *

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

if __name__ == "__main__":
    datMat = mat(loadDataSet('testSet.txt'))
    myCentroids, clustAssing = kMeans(datMat, 2)
    print('---')
    print(myCentroids)