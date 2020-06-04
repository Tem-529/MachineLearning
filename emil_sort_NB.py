#coding=UTF-8
import random
import numpy as np
import re
 
#解析英文文本，并返回列表
def textParse(bigString):
    #将单词以空格划分
    #listOfTokens = bigString.split()    
    #除去数字与字母以外的符号
    listOfTokens = re.split('\W+', bigString)
    #去除单词长度小于2的无用单词
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

#去列表中重复元素，并以列表形式返回
def createVocaList(dataSet):
    vocabSet = set({})
    #去重复元素，取并集
    
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
 
#统计每一文档（或邮件）在单词表中出现的次数，并以列表形式返回
def setOfWordsToVec(vocabList,inputSet,sel): 
    #创建0向量，其长度为单词量的总数
    returnVec = [0]*len(vocabList)
    #统计相应的词汇出现的数量
    for word in inputSet:
        if word in vocabList:
            if sel == 1:
                returnVec[vocabList.index(word)] += 1
            else:
                returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


 
#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory): 
    #获取训练文档数
    numTrainDocs = len(trainMatrix)
    #获取每一行词汇的数量
    numWords = len(trainMatrix[0])
    #计算垃圾邮件的比率  P(C1)
    pAbusive = sum(trainCategory)/float(numTrainDocs)  
    #统计非垃圾邮件中各单词在词数列表中出现的总数（向量形式）
    #拉普拉斯平滑，分子为1，分母为2
    p0Num = np.ones(numWords) 
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    #统计垃圾邮件总单词的总数（数值形式）
    p1Denom = 2.0
    for i in range(numTrainDocs):
        #如果是垃圾邮件
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom +=sum(trainMatrix[i])
        #如果是非垃圾邮件
        else:
            p0Num += trainMatrix[i]
            p0Denom +=sum(trainMatrix[i])
    #计算每个单词在垃圾邮件出现的概率（向量形式） P(xi|c1)
    p1Vect = p1Num/p1Denom
    #计算每个单词在非垃圾邮件出现的概率（向量形式）P(xi|c0)
    p0Vect = p0Num/p0Denom         
    return p0Vect,p1Vect,pAbusive
#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = 0
    p0 = 0
    for i in range(0,len(vec2Classify)):
        if vec2Classify[i] == 1:
            p1 += np.log(p1Vec[i])
            p0 += np.log(p0Vec[i])
        else: 
            p1 += np.log(1-p1Vec[i])
            p0 += np.log(1-p0Vec[i]) 
    p1 += np.log(pClass1)
    p0 += np.log(1.0 - pClass1)
    #p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
    #p0 = sum(vec2Classify*p0Vec)+np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else :
        return 0
#test
def spamtest():
    #导入并解析文本文件
    docList =[]
    classList=[]
    for i in range(26,51):
        #读取第i篇垃圾文件，并以列表形式返回
        #wordList = textParse(open('email/spam/{0}.txt'.format(i)).read())
        wordList = textParse(open('D:\python/train/spam/%d.txt' % i, 'r').read())
        #转化成二维列表
        docList.append(wordList)
        #标记文档为垃圾文档
        classList.append(1)
        #读取第i篇非垃圾文件，并以列表形式返回
        wordList = textParse(open('D:\python/train/ham/%d.txt' % i, 'r').read())
        #wordList = textParse(open('email/ham/{0}.txt'.format(i)).read())
        #转化成二维列表
        docList.append(wordList)
        #标记文档为非垃圾文档
        classList.append(0)
    #去除重复的单词元素
    #for i in range(1,26):
        #读取第i篇垃圾文件，并以列表形式返回
        #wordList = textParse(open('email/spam/{0}.txt'.format(i)).read())
        #wordList = textParse(open('D:\python/train/neg1/%d.txt' % i, 'r').read())
        #转化成二维列表
        #docList.append(wordList)
        #标记文档为垃圾文档
        #classList.append(1)
        #读取第i篇非垃圾文件，并以列表形式返回
        #wordList = textParse(open('D:\python/train/pos1/%d.txt' % i, 'r').read())
        #wordList = textParse(open('email/ham/{0}.txt'.format(i)).read())
        #转化成二维列表
        #docList.append(wordList)
        #标记文档为非垃圾文档
        #classList.append(0)
    #去除重复的单词元素
    vocabList = createVocaList(docList)
    #训练集，选40篇doc
    trainingSet = [x for x in range(50)]
    #测试集，选10篇doc
    testSet = []
    #50篇邮件随机选出10篇doc作测试集
    for i in range(10):
        #随机生成选取序号
        randIndex = int(random.uniform(0,len(trainingSet)))
        #测试集序号
        testSet.append(trainingSet[randIndex])
        #删除被选出序号    
        del trainingSet[randIndex]
    trainMat = [];trainClasses=[]
    #剩下40篇邮件作训练集
    for docIndex in trainingSet:
        trainMat.append(setOfWordsToVec(vocabList, docList[docIndex],1))
        #训练集邮件标签
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    #对测试集分类
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWordsToVec(vocabList,docList[docIndex],0)
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam)!=classList[docIndex]:
            errorCount+=1
            print("第%d封邮件分类错误：" % (docIndex), docList[docIndex])
    #print("错误率为：{0}".format(float(errorCount)/len(testSet)))
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))
    return float(errorCount) / len(testSet) * 100

if __name__ == '__main__':
    data = 0 
    for i in range(200):
        data +=spamtest()
    print('错误率：%.2f%%' % (float(data)/200))
