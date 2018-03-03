#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:38:53 2018

@author: f.ulloa10
"""
import random
import numpy as np
def sPix(fim,nIm,N):
    f1=[None]*(np.array(fim).shape[1])
    pix=[None]*(np.array(fim).shape[0])
    x=np.empty([N],dtype=int)
    y=np.empty([N],dtype=int)
    for b in range(N):
        x[b]=random.randint(0,np.array(fim[0][0]).shape[0])
        y[b]=random.randint(0,np.array(fim[0][0]).shape[1]/nIm)

    for i in range(np.array(fim).shape[0]):
        for j in range(np.array(fim).shape[1]):
            f1[j]=fim[i][j][x,y]
        pix[i]=f1
    return pix