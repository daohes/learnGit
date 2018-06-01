def loadDataSet():
	return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 提取数据集中所有元素并组成候选项集合
def createC1(dataSet):
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	return list(map(frozenset, C1))

# D 数据集， Ck候选项集， minSupport最小支持度
# 对数据集中各元素统计出现频次，除以数据集个数，得到各元素支持度
def scanD(D, Ck, minSupport):
	ssCnt = {}
	for tid in D:
		for can in Ck:
			if can.issubset(tid):
				if not can in ssCnt: 
					ssCnt[can] = 1
				else: ssCnt[can] += 1
	numItems = float(len(D))
	retList = [] 
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key]/numItems
		if support >= minSupport:
			retList.insert(0, key)
		supportData[key] = support
	return retList, supportData

if __name__ == "__main__":
	dataSet = loadDataSet()
	C1 = createC1(dataSet)
	D = list(map(set, dataSet))
	L1, suppData0 = scanD(D, C1, .5)
	print(L1)