from sklearn.linear_model import LogisticRegression

"""
function: 预测
modifytime：同上
"""
def colicSKlearn():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver= 'liblinear', max_iter=10).fit(trainingSet, trainingLabels)
    test_accuracy = classifier.score(testSet, testLabels) * 100
    print("正确率:%f%%" % test_accuracy)  

if __name__ == '__main__':
    colicSKlearn()

