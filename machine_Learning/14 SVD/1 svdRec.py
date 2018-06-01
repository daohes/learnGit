from numpy import *
from numpy import linalg as la

def loadExData():
	return [[1, 1, 1, 0, 0],
			[2, 2, 2, 0, 0],
			[1, 1, 1, 0, 0],
			[5, 5, 5, 0, 0],
			[1, 1, 0, 2, 2],
			[0, 0, 0, 3, 3],
			[0, 0, 0, 1, 1]]

'''
def loadExData():
	return [[2, 0, 0, 4, 4],
			[5, 5, 5, 3, 3],
			[2, 4, 2, 1, 2]]
'''

#欧氏距离相似度计算
def euclidSim(inA, inB):
	return 1.0/(1.0+la.norm(inA - inB))

#Pearson相似度计算
def perarsSim(inA, inB):
	if len(inA) < 3:
		return 1.
	return .5+.5*corrcoef(inA, inB, rowvar = 0)[0][1]

#余弦相似度计算
def cosSim(inA, inB):
	num = float(inA.T*inB)
	denom = la.norm(inA)*la.norm(inB)
	return .5+.5*(num/denom)


if __name__ == "__main__":
	myMat = mat(loadExData())
	print(euclidSim(myMat[:,0], myMat[:,4]))
	print(cosSim(myMat[:,0], myMat[:,4]))
	print(perarsSim(myMat[:,0], myMat[:,4]))
	

'''	U, Sigma, VT = linalg.svd(Data)
	print('Sigma:\n', Sigma, '\nU:\n', U, '\nVT:\n', VT)
	Sig3 = ([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
	print('Sig3:\n', Sig3)
	lowDData = matrix(U[:,:3]) * matrix(Sig3) * matrix(VT[:3,:])
	print('lowDData:\n',lowDData.A)
'''