import numpy as np
import operator

"""
classifier : KNN algorithm
"""

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1) # 各行元素相加
	distances = sqDistances ** .5
	sortedDistIndices = distances.argsort() 
	# [3,1,4,2] 的 argsort()结果是 [1,3,0,2]

	classCount = {}
	for i in range(k): # 指定取前k个
		voteILabel = labels[sortedDistIndices[i]]
		classCount[voteILabel] = classCount.get(voteILabel,0) + 1
		# get(key, default=None) 返回指定键的值
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetattr(1),reverse=True)
	#key=operator.itemgetter(1)根据字典的值进行排序,reverse降序排序字典
	return sortedClassCount[0][0]










