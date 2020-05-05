# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:33:31 2020

@author: Lenovo
"""
#knn算法实现mnist数据集识别，用训练集对测试集进行分类识别
 
import operator
import struct
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt


# 训练集
train_images_data = 'train-images.idx3-ubyte'
# 训练集标签
train_labels_data = 'train-labels.idx1-ubyte'
# 测试集
test_images_data = 't10k-images.idx3-ubyte'
# 测试集标签
test_labels_data = 't10k-labels.idx1-ubyte'

#解析图片
def decode_image(filename):
    bin_data = open(filename, 'rb').read()
    # 以二进制打开
    
    offset = 0
    fmt_header = '>iiii' 
    #大端转化4个int变量
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    #读取魔数，图片数量，图片高、宽
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    #偏移从像素值开始
    fmt_image = '>' + str(image_size) + 'B'
    #读取图片像素值共 784个字节
    
    images = np.empty((num_images, num_rows, num_cols))
    #创建数组保存图片信息
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #每张图片保存在28*28的数组中
        offset += struct.calcsize(fmt_image)
    print('解析完成')
    return images
 
 #解析标签
def decode_label(filename):
    bin_data = open(filename, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    #大端转化2个int变量
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    #魔数和图片数量
    
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    #标签为1字节
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        #标签保存至一维数组中
        offset += struct.calcsize(fmt_image)
    print('解析完成')
    return labels
 
 
def classify(test_data,train_data,labels,k):
    
    #第一维长度即训练集图片总数
    trainsize = train_data.shape[0]
    
    ###以下距离计算公式
    #测试图片纵向复制
    diffMat = np.tile(test_data,(trainsize,1))-train_data
    sqDiffMat = diffMat**2
    #行相加
    sqDistances = sqDiffMat.sum(axis=1) 
    distances = sqDistances ** 0.5
    
    
    #距离从小到大排序，返回图片序号
    sortedDistIndicies = distances.argsort()
    
    #创建空字典统计标签
    classCount = {}
    #前K个距离最小的标签存入字典，对应键值为标签出现次数
    for i in range(k):
        #获取标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #存入字典并出现记录次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        
    #按字典键值从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回出现次数最多的标签
    return sortedClassCount[0][0]

 
 
if __name__ == '__main__':
     
    #处理数据
    train_images = decode_image(train_images_data)
    train_labels = decode_label(train_labels_data)
    test_images = decode_image(test_images_data)
    test_labels = decode_label(test_labels_data)
    
    print('开始识别')

 
    m = 60000  # 创建一个读入数据的数组，保存图片信息
    trainingMat = np.zeros((m, 784))  # 初始化为零
 
    #平展成784维
    for i in range(m):
        for j in range(28):
            for k in range(28):
                trainingMat[i, 28*j+k] = train_images[i][j][k]
        
    
    mTest = 1000    
    #测试数量
    errorCount = 0.0
    #记录错误个数
    
    for i in range(mTest):
        classNumStr = test_labels[i]
        vectorUnderTest = np.zeros(784)
        #数组保存测试图片信息        
        for j in range(28):
            for k in range(28):
                vectorUnderTest[28*j+k] = test_images[i][j][k]  #存入第i个测试图
        
        #对测试图分类
        Result = classify(vectorUnderTest, trainingMat, train_labels, 3)
        print("识别结果：%d 正确答案：%d" % (Result, classNumStr))
        if (Result != classNumStr):
            errorCount += 1.0
            print("分类错误")
            
    print("\n错误数： %d" % errorCount)
    print("\n正确率率： %f" % ((mTest-errorCount) / float(mTest)))
    print ('数据处理结束')

