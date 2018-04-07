from skimage import color
from skimage.morphology import watershed
from scipy.io import loadmat
import scipy.ndimage.filters as filters
from sklearn.cluster import KMeans
from skimage.feature.peak import peak_local_max
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering


def segmentByClustering( rgbImage, featureSpace, clusteringMethod, numberOfClusters):
    
    def norIm(imageC):
        nIm = np.zeros((imageC.shape), dtype=np.uint8)
        cv2.normalize(imageC, nIm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return nIm
    
    if (featureSpace=='lab'):
        labIm=color.rgb2lab(rgbImage)
        feat=norIm(labIm)
        
    elif (featureSpace=='hsv'):
        hsvIm=color.rgb2hsv(rgbImage)
        feat=norIm(hsvIm)
        
    elif (featureSpace=='rgb+xy'):
        ySiz=rgbImage.shape[0]
        xSiz=rgbImage.shape[1]
        feat=np.zeros((ySiz,xSiz,5))
        x=np.tile(np.linspace(0,xSiz-1,xSiz),(ySiz,1))*255/(xSiz-1)
        y=np.transpose(np.tile(np.linspace(0,ySiz-1,ySiz),(xSiz,1)))*255/(ySiz-1)
        feat[:,:,0:3]=rgbImage
        feat[:,:,3]=x
        feat[:,:,4]=y
        
    elif (featureSpace=='lab+xy'):
        labIm=color.rgb2lab(rgbImage)
        labN=norIm(labIm)
        ySiz=labN.shape[0]
        xSiz=labN.shape[1]
        feat=np.zeros((ySiz,xSiz,5))
        x=np.tile(np.linspace(0,xSiz-1,xSiz),(ySiz,1))*255/(xSiz-1)
        y=np.transpose(np.tile(np.linspace(0,ySiz-1,ySiz),(xSiz,1)))*255/(ySiz-1)
        feat[:,:,0:3]=labN
        feat[:,:,3]=x
        feat[:,:,4]=y
        
    elif (featureSpace=='hsv+xy'):
        hsvIm=color.rgb2hsv(rgbImage)
        hsvN=norIm(hsvIm)
        ySiz=hsvN.shape[0]
        xSiz=hsvN.shape[1]
        feat=np.zeros((ySiz,xSiz,5))
        x=np.tile(np.linspace(0,xSiz-1,xSiz),(ySiz,1))*255/(xSiz-1)
        y=np.transpose(np.tile(np.linspace(0,ySiz-1,ySiz),(xSiz,1)))*255/(ySiz-1)
        feat[:,:,0:3]=hsvN
        feat[:,:,3]=x
        feat[:,:,4]=y
        
    else:
        feat=rgbImage
    
    if (clusteringMethod=='kmeans'):
        kmeans = KMeans(n_clusters=numberOfClusters, random_state=0, n_jobs=-1).fit(feat.reshape(-1,feat.shape[2]))
        labels=kmeans.predict(feat.reshape(-1,feat.shape[2]))
        segmentation=labels.reshape(feat.shape[0],feat.shape[1])
    
    elif(clusteringMethod=='gmm'):
        GMM=GaussianMixture(n_components=numberOfClusters).fit(feat.reshape(-1,feat.shape[2]))
        labels=GMM.predict(feat.reshape(-1,feat.shape[2]))
        segmentation=labels.reshape(feat.shape[0],feat.shape[1])
        
    elif(clusteringMethod=='hierarchical'):
        AC=AgglomerativeClustering(n_clusters=numberOfClusters).fit(feat.reshape(-1,feat.shape[2]))
        labels=AC.labels_
        segmentation=labels.reshape(feat.shape[0],feat.shape[1])
    
    elif(clusteringMethod=='watershed'):
        #Mean of the features
        nIm=np.zeros((feat.shape[0],feat.shape[1]))
        for i in range(feat.shape[2]):
            nIm=nIm+feat[:,:,i]
        nIm=nIm/feat.shape[2]
        #Estimate grad of image
        fx=cv2.Sobel(nIm,cv2.CV_64F,1,0,ksize=5)
        fy=cv2.Sobel(nIm,cv2.CV_64F,0,1,ksize=5)
        grad=np.sqrt((fx*fx+fy*fy))
        #Estimate connected components
        grad=norIm(grad)
        grad=cv2.threshold(grad,20,255,cv2.THRESH_BINARY)[1]
        grad[grad==255]=1
        grad=1-grad
        grad=np.array(grad,dtype=np.int8)
        ret, labels=cv2.connectedComponents(grad)
        unique, counts = np.unique(labels, return_counts=True)
        p=np.argsort(counts)[-numberOfClusters-1:]
        cnt=0
        pr=np.zeros((labels.shape[0],labels.shape[1],numberOfClusters+1))
        markers=np.zeros(labels.shape)
        for i in p:
            if (i!=0):
                pr[:,:,cnt]=labels
                pr[pr[:,:,cnt]!=i,cnt]=0
                pr[pr[:,:,cnt]==i,cnt]=cnt+1
            markers=markers+pr[:,:,cnt]
            cnt=cnt+1
            
        #Use watershed
        segmentation=watershed(grad,markers)
        
    return segmentation


