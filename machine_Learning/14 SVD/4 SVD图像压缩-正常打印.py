from numpy import *
from numpy import linalg as la

def transMat(inMat, thresh=.8):
	lowDMat = ''
	for i in range(32):
		for k in range(32):
			if float(inMat[i,k]) > thresh:
				lowDMat += '1'
			else:
				lowDMat += '0'
	return lowDMat

def printOri(fileName):
	f = open(fileName)
	print(f.read())

def imgCompress(numSV=3, thresh=.8):
	myl = []
	for line in open('0_5.txt').readlines():
		newRow = []
		for i in range(32):
			newRow.append(int(line[i]))
		myl.append(newRow)
	myMat = mat(myl)
	print('****original matrix****')
	printOri('0_5.txt')
	U, Sigma, VT = la.svd(myMat)
	SigRecon = mat(zeros((numSV, numSV)))
	for k in range(numSV):
		SigRecon[k,k] = Sigma[k]
	reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
	print('****reconstructed matrix using %d singular values****' % numSV)
	lowDMat = transMat(reconMat, thresh)
	for m in range(32):
		print(lowDMat[m*32:m*32+32])

if __name__ == "__main__":
	imgCompress(2)