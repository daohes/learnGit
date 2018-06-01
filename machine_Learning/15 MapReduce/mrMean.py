from mrjob.job import MRJob
from mrjob.step import MRStep
class MRmean(MRJob):
	def __init__(self, *args, **kwargs):
		super(MRmean, self).__init__(*args, **kwargs)
		self.inCount = 0
		self.inSum = 0
		self.inSqSum = 0

	def mapper(self, key, val):
		if False:
			yield
		inVal = float(val)
		self.inCount += 1
		self.inSum += inVal
		self.inSqSum += inVal * inVal

	def combiner(self):
		mn = self.inSum/self.inCount
		mnSq = self.inSqSum/self.inCount
		yield (1, [self.inCount, mn, mnSq])

	def reducer(self, key, packedValues):
		cumVal = 0.; cumSumSq = 0.; cumN = 0.
		for valArr in packedValues:
			nj = float(valArr[0])
			cumN += nj
			cumVal += nj*float(valArr[1])
			cumSumSq += nj*float(valArr[2])
		mean = cumVal/cumN
		var = (cumSumSq - 2*mean*cumVal + cumN*mean*mean)/cumN
		yield (mean, var)

	'''def steps(self):
		return ([MRStep(mapper=self.map, combiner=self.map_final, reducer=self.reduce)])
'''
if __name__ == "__main__":
	MRmean.run()