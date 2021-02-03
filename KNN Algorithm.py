#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math


def stdData(dataset):
    # copy the dataframe
    dataset_std = dataset.copy()
    # apply standardize formula
    for column in dataset.columns[:-1]:
        dataset_std[column] = (dataset_std[column] - dataset_std[column].mean()) / dataset_std[column].std()
    return dataset_std

def preprocessData(dataName):
    # read csv
    dataset = pd.read_csv(dataName)
    
    dataTemp = dataset.copy(deep=True)
    # replace 0 value with NaN for glucose, blood pressure, skin thickness, insulin, and BMI
    dataTemp[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dataTemp[['Glucose','BloodPressure',
                                                                          'SkinThickness','Insulin','BMI']].replace(0, np.NaN)
    
    # replace NaN value with median or mean for glucose, blood pressure, skin thickness, insulin, and BMI
    dataTemp['Glucose'].fillna(dataTemp['Glucose'].mean(), inplace = True)
    dataTemp['Insulin'].fillna(dataTemp['Insulin'].median(), inplace = True)
    dataTemp['BloodPressure'].fillna(dataTemp['BloodPressure'].mean(), inplace = True)
    dataTemp['SkinThickness'].fillna(dataTemp['SkinThickness'].median(), inplace = True)
    dataTemp['BMI'].fillna(dataTemp['BMI'].median(), inplace = True)
    
    # standardize dataset
    data = stdData(dataTemp)
    
    # create subsets
    # subset 1
    train1 = data[:614]
    test1 = data[614:]
    
    # subset 2
    temp1 = data.loc[:461]
    temp2 = data.loc[615:]
    train2 = pd.concat([temp1,temp2])
    test2 = data[461:615]

    # subset 3
    temp1 = data.loc[:307]
    temp2 = data.loc[462:]
    train3 = pd.concat([temp1,temp2])
    test3 = data[307:462]

    # subset 4
    temp1 = data.loc[:154]
    temp2 = data.loc[308:]
    train4 = pd.concat([temp1,temp2])
    test4 = data[154:308]
    
    # subset 5
    train5 = data[155:768]
    test5 = data[:155]

    dataTrain = [train1, train2, train3, train4, train5]
    dataTest = [test1, test2, test3, test4, test5]
    
    return dataTrain, dataTest
    

# calculate distance with euclidean distance
def calcDistance(x1, x2):
    temp = 0
    for i in range(len(x1) - 1):
        temp += math.pow(x1[i] - x2[i], 2)
        result = math.sqrt(temp)
    return result


def knn(k, dataTest, dataTrain):
    neighborDistance = []
    for j in range(len(dataTrain)):
        # calculate neighbor distances
        res = calcDistance(dataTest, dataTrain.iloc[j])
        neighborDistance.append([res, dataTrain.index[j]])    
        
    # sort the distance
    neighborDistance = sorted(neighborDistance)
    
    # pick k data
    kNN = neighborDistance[:k]
    
    # get label from k data
    for m in range(len(kNN)):
        if kNN[m][1] in dataTrain.index:
            labelTrain = dataTrain.loc[kNN[m][1]][-1]
            temp = kNN[m]
            temp.append(labelTrain)
    
    # determine classification
    counter1 = 0
    counter0 = 0
    label = 0
    for n in range(len(kNN)):
        if kNN[n][2] == 1:
            counter1 += 1
        else:
            counter0 += 1
    if counter1 > counter0:
        label = 1
        
    return label


# find best k value
def bestKFinder(accArray):
    temp = sorted(accArray, key=lambda x: x[1], reverse=True)
    return temp[0]


# calculate average accuracy of knn
def calcAvgAccuracy(accArray):
    temp = 0
    for data in accArray:
        temp += data
    return (temp / len(accArray))


# Main Program
dataTrain,dataTest = preprocessData('Diabetes.csv')
bestAccuracy = []

for a in range(len(dataTest)):
# for a in range(2):
    accResult = []
    print('total data subset ke-',a+1,' = ',len(dataTest[a]))
    i = 1
    while i < 50:
        accuracy = 0
        for j in range(len(dataTest[a])):
            pred = knn(i,dataTest[a].iloc[j],dataTrain[a])
#             print('pred=',pred)
#             print('actual=',dataTest[a].iloc[j][-1])
#             input('zzz..')
            if pred == dataTest[a].iloc[j][-1]:
                accuracy += 1
        print('k-',i,' correct = ',accuracy)
        accResult.append([i, (accuracy / len(dataTest[a])) * 100]) 
        i += 4
    bestK = bestKFinder(accResult)
    print('k terbaik adalah k = ',bestK[0],' dengan akurasi sebesar ',bestK[1])
    bestAccuracy.append(bestK[1]) 
avgAccuracy = calcAvgAccuracy(bestAccuracy)
print('rata-rata akurasi k terbaik dari 5 fold cross-validation adalah ', avgAccuracy)

