# -*- coding: UTF-8 -*-
import os
import random
import jieba

"""
function: 中文文本处理
modify time:20180325
"""
def TextProcessing(folder_path, test_size = 0.2):
    folder_list = os.listdir(folder_path)                        #查看folder_path下的文件
    data_list = []                                                #训练集
    class_list = []

    #遍历每个子文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)        #根据子文件夹，生成新的路径
        files = os.listdir(new_folder_path)                        #存放子文件夹下的txt文件的列表

        j = 1
        #遍历每个txt文件
        for file in files:
            if j > 100:                                            #每类txt样本数最多100个
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:    #打开txt文件
                raw = f.read()

            word_cut = jieba.cut(raw, cut_all = False)            #精简模式，返回一个可迭代的generator
            word_list = list(word_cut)                            #generator转换为list

            data_list.append(word_list)
            class_list.append(folder)
            j += 1

    data_class_list = list(zip(data_list, class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    all_words_tuple_list = sorted(all_words_dict.items(),\
                                  key = lambda f:f[1], reverse = True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)
    all_words_list = list(all_words_list)
    return all_words_list, train_data_list, test_data_list,\
           train_class_list, test_class_list

"""
function: 读取文件里的内容，并去重
modifytime: 20180325
"""
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, "r", encoding = 'utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                words_set.add(word)
    return words_set

"""
函数说明:文本特征选取
modifytime: 20180325
"""
def words_dict(all_words_list, deleteN, stopwords_set = set()):
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list)):
        if n > 1000:
            break
        if not all_words_list[t].isdigit() \
           and all_words_list[t] not in stopwords_set \
           and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words

if __name__ == "__main__":
    folder_path = './SogouC/Sample'
    all_words_list, train_data_list, test_data_list,\
                    train_class_list, test_class_list \
                    = TextProcessing(folder_path, test_size=0.2)
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    feature_words = words_dict(all_words_list, 100, stopwords_set)
    print(feature_words)
    
