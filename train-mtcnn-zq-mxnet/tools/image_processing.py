import numpy as np
from numpy import *
import mxnet.ndarray as nd
import cv2
import math
import sys,os
sys.path.append(os.getcwd())
from config import config

def transform(im, train = False):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :return: [batch, channel, height, width]
    """
    if train:
        scale = np.random.randint(3,7) * 0.2
        im = im * scale
        if config.enable_gray:
            gray_flag = np.random.randint(0,2)
            if gray_flag == 1:
                #im[:,:,:] = np.sum(im * [0.114,0.587,0.299], axis=-1, keepdims=True)
                gray_im = im[:,:,0]*0.114+im[:,:,1]*0.587+im[:,:,2]*0.299
                im[:,:,0] = gray_im
                im[:,:,1] = gray_im
                im[:,:,2] = gray_im
                
		
    im_tensor = im.transpose(2, 0, 1)
    im_tensor = im_tensor[np.newaxis, :]
    im_tensor = (im_tensor - 127.5)*0.0078125
    return im_tensor

def SaltAndPepper(src,percetage):  
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1]) 
    for i in range(SP_NoiseNum): 
        randR=np.random.randint(0,src.shape[0]-1) 
        randG=np.random.randint(0,src.shape[1]-1) 
        randB=np.random.randint(0,3)
        if np.random.randint(0,1)==0: 
            SP_NoiseImg[randR,randG,randB]=0 
        else: 
            SP_NoiseImg[randR,randG,randB]=255 
    return SP_NoiseImg 
def addGaussianNoise(image,percetage): 
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1]) 
    for i in range(G_NoiseNum): 
        temp_x = np.random.randint(0,h) 
        temp_y = np.random.randint(0,w) 
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0] 
    return G_Noiseimg
#dimming
def darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image_copy
def rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image
	
def rotateWithLandmark(image, landmark, angle, scale):
    if angle == 0:
        rot_image = image.copy()
        landmark1 = landmark.copy()
        return rot_image,landmark1
    else:
        w = image.shape[1]
        h = image.shape[0]
        cx = landmark[4]
        cy = landmark[5]
        #rotate matrix
        M = cv2.getRotationMatrix2D((cx,cy), angle, scale)
    
    
        in_coords = np.array([[landmark[0], landmark[2], landmark[4], landmark[6], landmark[8]], 
                              [landmark[1], landmark[3], landmark[5], landmark[7], landmark[9]], 
                              [1,1,1,1,1]], dtype=np.float32)
    
        #rotate
  
        rot_image = cv2.warpAffine(image,M,(w,h))
        out_coords = np.dot(M,in_coords)
        landmark1 = np.array(landmark,dtype=np.float32).copy()
        for i in range(5):
            landmark1[i*2] = out_coords[0][i]
            landmark1[i*2+1] = out_coords[1][i]
        return rot_image, landmark1