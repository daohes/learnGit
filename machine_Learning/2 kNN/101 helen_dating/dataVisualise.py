from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
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

def showdatas(datingDataMat, datingLabels):
	font = FontProperties(fname="c:/windows/fonts/simkai.ttf",size=14)

	fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False,figsize=(13,8))

	numberOfLabels = len(datingLabels)
	LabelsColors = []
	for i in datingLabels:
		if i == 1:
			LabelsColors.append('black')
		if i == 2:
			LabelsColors.append('orange')
		if i == 3:
			LabelsColors.append('red')

	axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
	# alpha 表示透明度
	axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
	axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
	axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
	plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
	plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
	plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
	axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
	#设置标题,x轴label,y轴label
	axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
	axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
	axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
	plt.setp(axs1_title_text, size=9, weight='bold', color='red') 
	plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black') 
	plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
	axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
	axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
	axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
	axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
	plt.setp(axs2_title_text, size=9, weight='bold', color='red') 
	plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black') 
	plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

	didntLike = mlines.Line2D([], [], color='black', marker='.',\
    	markersize=6, label='不喜欢')
	smallDoses = mlines.Line2D([], [], color='orange', marker='.',\
    	markersize=6, label='一般般')
	largeDoses = mlines.Line2D([], [], color='red', marker='.',\
    	markersize=6, label='很喜欢')

	axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
	plt.show()


if __name__ == "__main__":
	filename = "datingTestSet.txt"
	datingDataMat, datingLabels = file2matrix(filename)
	showdatas(datingDataMat, datingLabels)