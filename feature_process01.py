#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:27:56 2017

@author: root
"""

import numpy as np
import math
import matplotlib.pyplot as plt
file_path1="encode_feature_set/LOGO_true.txt"
file_path2="encode_feature_set/LOGO_train_Less_5.txt"
file_path3="encode_feature_set/LOGO_train_Less_10.txt"
file_path4="encode_feature_set/LOGO_train_Less_15.txt"
file_path5="encode_feature_set/LOGO_train_Less_20.txt"
file_path6="encode_feature_set/LOGO_train_More_5.txt"
file_path7="encode_feature_set/LOGO_train_More_10.txt"
file_path8="encode_feature_set/LOGO_train_More_15.txt"
file_path9="encode_feature_set/LOGO_train_More_20.txt"
file_path10="qt_feature.txt"
def distanceMat1(inX,dataSet):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    sqDistances=sqDistances**0.5
    distance_mean=sqDistances.sum()/dataSetSize
    return distance_mean


def distanceMat2(inX,dataSet):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    sqDistances=sqDistances**0.5
    distance_max=sqDistances.max()
    return distance_max

def distanceMat3(inX,dataSet):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    sqDistances=sqDistances**0.5
    distance_min=sqDistances.min()
    return distance_min


def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet=normDataSet/(np.tile(ranges,(m,1)))
    return normDataSet

def mean_vector(dataSet):
    n=dataSet.shape[0]
    sum_col=dataSet.sum(axis=0)
    mean_col=sum_col/n
    return mean_col

def PCAchangeMat(dataSet):
    real_vector = dataSet.reshape(650,256)
    real_vector_trans = real_vector.T
    covariance = np.dot(real_vector_trans, real_vector)

    eigvalues, eigvects = np.linalg.eig(covariance)

    eigIndex = np.argsort(eigvalues)
    n_eigValIndex = eigIndex[-1:-(13+1):-1]
    n_eigVect = eigvects[:, n_eigValIndex]
    return n_eigVect

def gaussianMat(inX,dataSet1):
    n_eigVect=PCAchangeMat(dataSet1)
    dataSet=np.dot(dataSet1,n_eigVect)
    featuresize=dataSet.shape[1]
    cov_Mat=np.cov(dataSet.T)
    cov_value=np.linalg.det(cov_Mat)
    print (cov_value)
    temp1=np.dot((inX-mean_vector(cov_Mat)),cov_Mat)
    temp2=np.dot(temp1,(inX-mean_vector(cov_Mat)))
    res1=1/math.sqrt(math.pow(2*math.pi,featuresize)*cov_value)
    print (res1)
    res2=math.exp(-1/2*temp2)
    print (res2)
    return res1*res2

#file_path1='image_vect_feature_OK2695.txt'
#file_path2='image_vect_feature_NG2695.txt'
#file_path1='OK15wuquanlianjie.txt'
#file_path2='NG15wuquanlianjie.txt'
#img_vec=np.loadtxt(file_path1)
#img_vec_OK=img_vec[0:127,:]
#img_vec_OK_test=img_vec[127:,:]
#img_vec_NG_test=np.loadtxt(file_path2)
#n_OK_test=img_vec_OK_test.shape[0]
#n_NG_test=img_vec_NG_test.shape[0]
#distance_OK=np.zeros(n_OK_test)
#distance_NG=np.zeros(n_NG_test)
#
#for i in range(21):
#    distance_OK[i]=distanceMat1(img_vec_OK_test[i],img_vec_OK)
#    distance_NG[i]=distanceMat1(img_vec_NG_test[i],img_vec_OK)
#    
#x=np.arange(1,22,1)
#plt.figure()
#plt.plot(x,distance_OK,label='OK',color='blue')
#plt.plot(x,distance_NG,label='NG',color='red')
#plt.xlim(0,21)
#plt.xlabel('sample')
#plt.ylabel('distance')
#plt.legend()
#plt.title('distance_mean')
#plt.show()

#img_vec1=np.loadtxt(file_path1)
#img_vec2=np.loadtxt(file_path9)
#img_vec=np.loadtxt(file_path10)
#img_vec1=img_vec[0:25]
#img_vec2=img_vec[25:50]
#print (img_vec1.shape)
#print (img_vec2.shape)
#n_OK_test=img_vec1.shape[0]
#n_NG_test=img_vec2.shape[0]
#distance_OK=np.zeros(n_OK_test)
#distance_NG=np.zeros(n_NG_test)
#
#for i in range(25):
#    distance_OK[i]=distanceMat1(img_vec1[i],img_vec1)
#    distance_NG[i]=distanceMat1(img_vec2[i],img_vec1)
#    
#x=np.arange(1,26,1)
#plt.figure()
#plt.plot(x,distance_OK,label='OK',color='blue')
#plt.plot(x,distance_NG,label='NG',color='red')
#plt.xlim(0,25)
#plt.xlabel('sample')
#plt.ylabel('distance')
#plt.legend()
#plt.title('distance_mean')
#plt.show()
#num=50
#plt.figure()
#histo=plt.hist(distance_OK,num,color="blue")
#plt.plot(histo[1][0:num],histo[0],"r")
#a=histo[0]
#print(np.where(a=0))
#plt.show()
#
#file_path1='image_vector.txt'
#file_path2='image_vector_NG2_700.txt'
#img_vec1=np.loadtxt(file_path1)
#img_vec2=np.loadtxt(file_path2)
#print (img_vec1.shape)
#print (img_vec2.shape)
#for i in img_vec1[0]:
#    print (i)
#print ("*************************")
#for i in img_vec1[1]:
#    print (i)