# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:53:31 2020

@author: Lenovo
"""


#!usr/bin/env python
 
import importlib,sys
import os
import struct
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
from kernel_knn import KernelKNN 
importlib.reload(sys)
 
# 贝叶斯分类器
def naive_bayes_classifier(train_x, train_y):
    #from sklearn.naive_bayes import GaussianNB
    #model = GaussianNB()
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=1)
    model.fit(train_x, train_y)
    return model
 
# KNN分类器
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_x, train_y)
    return model

#Kernel-KNN
def mnist_simulation(train_x,train_y,test_x,test_y,k_value,kernel_param):
    
    kernel_k_NN_mnist = KernelKNN(k_value)    
    kernel_k_NN_mnist.training(train_x,train_y)
    predicted_mnist = kernel_k_NN_mnist.prediction(test_x, kernel_param)
    #accuracy = kernel_k_NN_mnist.evaluation_prediction(predicted_mnist, test_y)
    accuracy = metrics.accuracy_score(test_y, predicted_mnist)
    print ('accuracy: %.2f%%' % (100 * accuracy))
    return accuracy
 
# 逻辑回归分类器
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=0.35,penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=1000)
    model.fit(train_x, train_y)
    return model
  
#SVM分类器
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(C=225,kernel='rbf', probability=False,gamma='auto')
    model.fit(train_x, train_y)
    return model
 
# SVM最优参数
def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid,cv=4)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_params_best_params_
    for para, val in best_parameters.items():
        print (para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=False)
    model.fit(train_x, train_y)
    return model
 
#搜素LR最优参数    
def LRCV(train_x,train_y,test_x,test_y): 
    from sklearn.linear_model import LogisticRegression
    param_grid = {"max_iter":[50,100,200,300,400, 500],
             "C":[0.001,0.01,0.1,1,10,50,100  ]}
    lr = LogisticRegression(penalty='l2',multi_class='multinomial', solver='lbfgs')
    #交叉验证搜索参数
    grid_search = GridSearchCV(lr,param_grid,cv=5)
    grid_search.fit(train_x,train_y)
    print("Test set score:{:.2f}".format(grid_search.score(test_x,test_y)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

#读取数据
def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f,encoding='bytes')
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x, train_y, test_x, test_y

#读取图片
def read_image(file_name):
    #先用二进制方式把文件都读进来
    file_handle=open(file_name,"rb")  #以二进制打开文档
    file_content=file_handle.read()   #读取到缓冲区中
    offset=0
    head = struct.unpack_from('>IIII', file_content, offset)  # 取前4个整数，返回一个元组
    offset += struct.calcsize('>IIII')
    imgNum = head[1]  #图片数
    rows = head[2]   #宽度
    cols = head[3]  #高度
    images=np.empty((imgNum , 784))#empty，是它所常见的数组内的所有元素均为空，没有实际意义，它是创建数组最快的方法
    image_size=rows*cols#单个图片的大小
    fmt='>' + str(image_size) + 'B'#单个图片的format

    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        # images[i] = np.array(struct.unpack_from(fmt, file_content, offset)).reshape((rows, cols))
        offset += struct.calcsize(fmt)
    return images

#读取标签
def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中

    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')

    labelNum = head[1]  # label数
    # print(labelNum)
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)

#[0,1]区间归一化
def normalize(data):
    m=data.shape[0]
    n=np.array(data).shape[1]
    for i in range(m):
        for j in range(n):
            data[i,j]=float(data[i,j])/255
    return data

#二值化
def  normalize_new(data):#图片像素二值化，变成0-1分布
    m=data.shape[0]
    n=np.array(data).shape[1]
    for i in range(m):
        for j in range(n):
            if data[i,j]!=0:
                data[i,j]=1
            else:
                data[i,j]=0
    return data

#加载数据
def loadDataSet():
    train_x_filename="D:/python/train-images.idx3-ubyte"
    train_y_filename="D:/python/train-labels.idx1-ubyte"
    test_x_filename="D:/python/t10k-images.idx3-ubyte"
    test_y_filename="D:/python/t10k-labels.idx1-ubyte"
    train_x=read_image(train_x_filename)#训练集特征 60000*784 的矩阵
    train_y=read_label(train_y_filename)#训练集标签 60000*1的矩阵
    test_x=read_image(test_x_filename)#测试集特征  10000*784的矩阵
    test_y=read_label(test_y_filename)#测试集标签 10000*1的矩阵
    
    #特征归一化
    #可以比较这两种预处理的方式最后得到的结果
    train_x = normalize(train_x)
    test_x = normalize(test_x)

    #train_x=normalize_new(train_x)
    #test_x=normalize_new(test_x)

    return train_x, test_x, train_y, test_y



     
if __name__ == '__main__':
        
    score= 0.0 #模型评估得分
    accuracy1 = 0.0 #准确率
    select = 0   #选择数据集，0为完整数据集，1为部分数据集
    model_save_file = None
    model_save = {}
    #分类器列表
    test_classifiers = ['NB','KNN','LR','SVM']
    classifiers = {'NB':naive_bayes_classifier,
                  'KNN':knn_classifier,
                   'LR':logistic_regression_classifier,
                  'SVM':svm_classifier,
                'SVMCV':svm_cross_validation
    }
    #核函数列表
    kernel_list = ['linear','polynomial','gaussian']
    print ('reading training and testing data...')
    #读取数据集
    train_x, test_x,train_y, test_y = loadDataSet()
    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    print ('******************** Data Info *********************')
    print ('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))

    for classifier in test_classifiers:
        print ('******************* %s ********************' % classifier)
        if select == 1:
            x_train, x_train_s, y_train, y_train_s = train_test_split( train_x, train_y,test_size=0.1, random_state=0)
            x_test, x_test_s ,y_test,y_test_s = train_test_split( test_x, test_y, test_size=0.1 , random_state=0)
            print(len(x_train_s))
            start_time = time.time()
            model = classifiers[classifier](x_train_s, y_train_s)
            print ('training took %fs!' % (time.time() - start_time))
            start_time = time.time()
            predict = model.predict(x_test_s)
            print('test took %fs!' % (time.time() - start_time))
            accuracy = metrics.accuracy_score(y_test_s, predict)
            print ('accuracy: %.2f%%' % (100 * accuracy))
            for i in range(1,5):
                testsize=0.2*i
                x_train, x_train_s, y_train, y_train_s = train_test_split( train_x, train_y,test_size=testsize, random_state=0)
                x_test, x_test_s ,y_test,y_test_s = train_test_split( test_x, test_y, test_size=testsize , random_state=0)
                print(len(x_train_s))
                start_time = time.time()
                model = classifiers[classifier](x_train_s, y_train_s)
                print ('training took %fs!' % (time.time() - start_time))
                start_time = time.time()
                predict = model.predict(x_test_s)
                print('test took %fs!' % (time.time() - start_time))
                accuracy = metrics.accuracy_score(y_test_s, predict)
                print ('accuracy: %.2f%%' % (100 * accuracy))
        else:
            start_time = time.time()
            model = classifiers[classifier](train_x, train_y)
            print ('training took %fs!' % (time.time() - start_time))
            start_time = time.time()
            predict = model.predict(test_x)
            print('test took %fs!' % (time.time() - start_time))
            accuracy = metrics.accuracy_score(test_y, predict)
            print ('accuracy: %.2f%%' % (100 * accuracy))
            model_save[classifier] = model
    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))
    print ('******************* kernel_knn********************' )
    #对完整数据集测试每个核函数
    #for kernel_param in kernel_list:
    #    print('*****%s*******' % kernel_param)
    #    x_train, x_train_s, y_train, y_train_s = train_test_split( train_x, train_y,test_size=0.1, random_state=0)
    #    x_test, x_test_s ,y_test,y_test_s = train_test_split( test_x, test_y, test_size=0.1 , random_state=0)
    #    print(len(x_train_s))
    #    start_time = time.time()
    #    mnist_simulation(x_train_s, y_train_s,x_test_s,y_test_s,3,kernel_param)
    #    print('kernel knn cost %fs!'  % (time.time() - start_time))
    #    for k in range(1,5):
    #        testsize=0.2*k
    #        x_train, x_train_s, y_train, y_train_s = train_test_split( train_x, train_y,test_size=testsize, random_state=0)
    #        x_test, x_test_s ,y_test,y_test_s = train_test_split( test_x, test_y, test_size=testsize , random_state=0)
    #        print(len(x_train_s))
    #        start_time = time.time()
    #        mnist_simulation(x_train_s, y_train_s,x_test_s,y_test_s,3,kernel_param)
    #    start_time = time.time()
    #    mnist_simulation(train_x, train_y,test_x,test_y,3,kernel_param)
    #    print('kernel knn cost %fs!'  % (time.time() - start_time))
    
    #Kernel-KNN
    if select == 1:
        x_train, x_train_s, y_train, y_train_s = train_test_split( train_x, train_y, test_size=0.4, random_state=0)
        x_test, x_test_s ,y_test,y_test_s = train_test_split( test_x, test_y, test_size=0.4, random_state=0)
        start_time = time.time()
        mnist_simulation(x_train_s, y_train_s,x_test_s,y_test_s,3,kernel_list[0])
        print('kernel knn cost %fs!'  % (time.time() - start_time))
    else:
        start_time = time.time()
        mnist_simulation(train_x, train_y,test_x,test_y,3,kernel_list[0])
        print('kernel knn cost %fs!'  % (time.time() - start_time))
