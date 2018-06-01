# -*- coding: UTF-8 -*-

'''
function: 创建实验样本
modify time:
2018-03-21
'''

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], \
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],\
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],\
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],\
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],\
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

if __name__ == "__main__":
    postingList, classVec = loadDataSet()
    for each in postingList:
        print(each)
    print(classVec)
