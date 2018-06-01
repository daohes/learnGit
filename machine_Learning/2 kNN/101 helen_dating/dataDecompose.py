import numpy as np

def file2matrix(filename):
	fr = open(filename)
	arrayOlines = fr.readlines()
	numberOfLines = len(arrayOlines)
	returnMat = np.zeros((numberOfLines,3))
	classLabelVector = []
	index = 0

	for line in arrayOlines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		if listFromLine[-1] == "didntLike":
			classLabelVector.append(1)
		elif listFromLine[-1] == "smallDoses":
			classLabelVector.append(2)
		else:
			classLabelVector.append(3)
		index += 1
	return returnMat, classLabelVector

if __name__ == "__main__":
	filename = "datingTestSet.txt"
	datingDataMat, datingLabels = file2matrix(filename)
	print(datingDataMat)
	print(datingLabels)