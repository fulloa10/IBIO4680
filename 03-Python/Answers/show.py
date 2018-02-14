#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:17:02 2018

@author: f.ulloa10
"""
from PIL import Image
from scipy.io import loadmat
import scipy as sc
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import time

if (os.path.isdir("BSR")==False):
    os.system("wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz")
    os.system("tar -xvf BSR_full.tgz")
    
start_time = time.time()

trainPath= "BSR/BSDS500/data/images/train"
GTPath= "BSR/BSDS500/data/groundTruth/train"
names=os.listdir(trainPath)

fig=plt.figure(figsize=(13, 10))
columns = 4
rows = 3
nm=[]

for i in range(1, columns+1):
    j=random.randint(0,201);
    names2=names[j]
    names2=names2.replace(".jpg",".mat")
    img = Image.open(os.path.join(trainPath,names[j]))
    mat = loadmat(os.path.join(GTPath,names2), squeeze_me=True)
    seg=mat["groundTruth"]
    seg=seg[0]
    seg1=seg.item(0)
    seg=seg1[0]
    boun=seg1[1]
    seg=sc.misc.imresize(seg,(256,256),interp="cubic")
    boun=sc.misc.imresize(boun,(256,256),interp="cubic")
    boun=np.invert(boun)
    img=img.resize((256,256),Image.ANTIALIAS)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    
    fig.add_subplot(rows, columns, i+columns)
    plt.imshow(seg)
    plt.xticks([])
    plt.yticks([])
    
    fig.add_subplot(rows, columns, i+2*columns)
    plt.imshow(boun, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    
    nm.append(names[j] + " " + names2)

plt.savefig("imagen.jpg")
plt.show()

with open("names.txt", "w") as output:
    output.write(str(nm))

elapsed_time = time.time() - start_time
print(elapsed_time)