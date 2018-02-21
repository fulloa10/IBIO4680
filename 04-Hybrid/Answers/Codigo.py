# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:23:50 2018

@author: Federico Ulloa
"""

import scipy as sc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2


im1=ndimage.imread("Imagenes/Mario.jpeg")
im2=ndimage.imread("Imagenes/Oscar.JPG")

im1=sc.misc.imresize(im1,(800,704),interp="cubic")
im2=sc.misc.imresize(im2,(800,704),interp="cubic")

im1_bajo=ndimage.gaussian_filter(im1,5)
im2_bajo=ndimage.gaussian_filter(im2,15)

im2_alto=cv2.subtract(im2,im2_bajo)
im=cv2.add(im2_alto,im1_bajo)
plt.imshow(im2_alto)
plt.imshow(im)
plt.imsave("Hybrid.png",im)

Co = im1.copy()
pIm1 = [Co]
for i in range(6):
    Co = cv2.pyrDown(Co)
    pIm1.append(Co)
    name="PG1_Layer"+str(i)+".png"
    plt.imsave(name,Co)

plIm1 = [pIm1[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(pIm1[i])
    L = cv2.subtract(pIm1[i-1],GE)
    plIm1.append(L)

Co = im2.copy()
pIm2 = [Co]
for i in range(6):
    Co = cv2.pyrDown(Co)
    pIm2.append(Co)
    name="PG2_Layer"+str(i)+".png"
    plt.imsave(name,Co)

plIm2 = [pIm2[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(pIm2[i])
    L = cv2.subtract(pIm2[i-1],GE)
    plIm2.append(L)

LS = []
for la,lb in zip(plIm1,plIm2):
    rows,cols,dpt = la.shape
    mit=int(cols/2)
    ls = np.hstack((la[:,0:mit], lb[:,mit:]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

plt.imsave("Blending.png",ls_)
