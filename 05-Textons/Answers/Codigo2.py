#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:01:18 2018

@author: Federico Ulloa
"""
import sys
import time
import numpy as np
from skimage import color
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import cv2

sys.path.append('lib/python')

from fbCreate import fbCreate
from fbRun import fbRun
from computeTextons import computeTextons
from assignTextons import assignTextons

start_time = time.time()

#Initialize Var
imCon=np.empty([480,640])
imSize=100
nIm=0

#Create a filter bank with deafult params
fb = fbCreate()

#Load and stack train images
path='train/'
for folder in os.listdir(path):
    for img in os.listdir(os.path.join(path,folder)):
        image=color.rgb2gray(io.imread(os.path.join(path,folder,img)))
        image=cv2.resize(image,(imSize,imSize))
        if nIm==0:
            imCon=image
        else:
            imCon=np.hstack((imCon,image))
        nIm=nIm+1;
print('1')

#Set number of clusters
k = 40

#Apply filterbank to sample image
filterResponses = fbRun(fb,imCon)
print('2')

# Compute textons for filter response
map, textons = computeTextons(filterResponses,k)
print('3')

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

## Assign textons to the train images
X=np.zeros([nIm,k])
c=0;
y=np.zeros([nIm])
la=0;
print('4')

for folder in os.listdir(path):
    for img in os.listdir(os.path.join(path,folder)):
        image=color.rgb2gray(io.imread(os.path.join(path,folder,img)))
        image=cv2.resize(image,(imSize,imSize))
        tmap = assignTextons(fbRun(fb,image),textons.transpose())
        X[c,:]=histc(tmap.flatten(), np.arange(k))/tmap.size
        y[c]=la
        c=c+1;
    la=la+1;
    print(la)

#Load test images
path='test/'
la=0
c=0
yTes=np.zeros([250])
xTes=np.zeros([250,k])
for folder in os.listdir(path):
    for img in os.listdir(os.path.join(path,folder)):
        imageTes=color.rgb2gray(io.imread(os.path.join(path,folder,img)))
        imageTes=cv2.resize(imageTes,(imSize,imSize))
        tmapTes = assignTextons(fbRun(fb,imageTes),textons.transpose())
        xTes[c,:]=histc(tmapTes.flatten(), np.arange(k))/tmapTes.size
        yTes[c]=la
        c=c+1
    la=la+1;
    print(la)

#Predict with KNN
nK=np.array([1,3,5,7,9,11,13,15,17,19,21,23,25,27,30])
acaKNN=np.zeros([nK.size])
acaTKNN=np.zeros([nK.size])
con=0
for n_ne in nK:
    neigh=KNeighborsClassifier(n_ne)
    neigh.fit(X,y)
    yPre=neigh.predict(xTes)
    yTPre=neigh.predict(X)
    cM=confusion_matrix(yTes,yPre)
    acaKNN[con]=accuracy_score(yPre,yTes)
    acaTKNN[con]=accuracy_score(yTPre,y)
    con+=1

plt.figure()
plt.plot(nK,acaKNN)
plt.xlabel('Numero de vecinos mas cercanos')
plt.ylabel('ACA(%)')
plt.title('Indicador ACA para diferentes K')
plt.grid()
plt.savefig('ACAKNN')
#Predict with Random Forest
dep=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,25,27,30])
acaRF=np.zeros([dep.size])
acaTRF=np.zeros([dep.size])
con=0
for mxD in dep:
    rF=RandomForestClassifier(max_depth=mxD)
    rF.fit(X,y)
    yPreRF=rF.predict(xTes)
    yTPreRF=rF.predict(X)
    cMRF=confusion_matrix(yTes,yPreRF)
    acaRF[con]=accuracy_score(yPreRF,yTes)
    acaTRF[con]=accuracy_score(yTPreRF,y)
    con+=1

plt.figure()
plt.plot(dep,acaRF)
plt.xlabel('Profundidad maxima')
plt.ylabel('ACA(%)')
plt.title('Indicador ACA para diferentes profundidades')
plt.grid()
plt.savefig('ACARF')

elapsed_time = time.time() - start_time
print(elapsed_time)
