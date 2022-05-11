import scipy.sparse as sp
import os
import math
import random
import numpy as np
from scipy.io import loadmat
import operator




# 读取数据，返回样本与标签
def load_data(folder_path, frequency):

    emotionData = [] # 存储每一个subject的数组
    emotionLabel = []
        
    emotion = [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0] #情绪标签序列，每一个mat文件的情绪标签顺序相同

    for i in range(15):
        dataKey =frequency + str(i+1) #读取的数据对应的Key

        metaData = np.array((loadmat(folder_path,verify_compressed_data_integrity=False)[dataKey])).astype('float') #读取到原始的三维元数据

        trMetaData = np.swapaxes(metaData,0,1) #(235,62,5)
        subArrayLength = np.shape(trMetaData)[0] #读取每一个trial的时间量（非固定值）
        trMetaData = np.array(trMetaData) 

        emotionData.append(trMetaData)
        emotionLabel.append([emotion[i],]*subArrayLength) #对情绪数据按秒打上标签
    return emotionData, emotionLabel


#对读取到的数据进行变形，裁切和电极重新排序
def data_crop_and_sort(sample, label):
    """
    @input:
    sample:[trail, batch, pole, fre] 
    label: [trail, batch] 

    @output:
    _sample:[batch, channel, pole, fre]
    _label: [batch,] 

    """
    #读数据进行变形，输出为[batchs, channel, pole, fre] 
    sample_train, label_train, sample_test, label_test = [], [], [], []
    sample_np = np.array([])

    for i in range(len(sample)):
        y = label[i]
        x = sample[i]

        if i < 9:
            if len(sample_train) != 0:
                sample_train=np.concatenate((sample_train, x),axis=0) # 拼接矩阵 
                label_train=np.hstack((label_train,y))     
            else:
                sample_train = x
                label_train = y
        else:
            if len(sample_test) != 0:
                sample_test=np.concatenate((sample_test,x),axis=0)
                label_test=np.hstack((label_test,y))   
            else:
                sample_test = x
                label_test = y
			
    sample_train_np = np.array(sample_train)
    label_train_np = np.array(label_train)
    sample_test_np = np.array(sample_test)
    label_test_np = np.array(label_test)


    #对数据根据空间结构进行重新排序,与邻接矩阵对齐
    sample_train = np.concatenate((sample_train_np[:,3:4], sample_train_np[:,0:3],sample_train_np[:,4:]), axis = 1)
    sample_test = np.concatenate((sample_test_np[:,3:4], sample_test_np[:,0:3],sample_test_np[:,4:]), axis = 1)
    return sample_train, label_train_np, sample_test, label_test_np



def extend_normal(sample):
    """
    @input:sample under normalized
    @output:sample normalized
    """
    for i in range(len(sample)):

        max_sample = np.max(sample[i],axis=0)
        min_sample = np.min(sample[i],axis=0)
        dif_sample = max_sample - min_sample

        sample[i] = (sample[i] -  min_sample) / dif_sample

    return sample


def data_intra_sub_single_fre(data_path, frequency):


    X_, y_ = load_data(data_path, frequency)
    X_train, y_train, X_test, y_test = data_crop_and_sort(X_, y_)

    X_test = extend_normal(X_test)
    X_train = extend_normal(X_train) 

    X_train = np.array(X_train)       
    X_test = np.array(X_test)         
    y_train = np.array(y_train)        
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test



