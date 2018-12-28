# conding:utf-8

import mxnet as mx
import numpy as np

import cv2

rgb_mean = mx.nd.array([0.485, 0.456, 0.406])
rgb_std = mx.nd.array([0.229, 0.224, 0.225])

# normalize
def normalize_image(data):
    """
    Info: Data normalize.
    """
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std

# resize
class Resize:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        
    def __call__(self, img, lbl):
        img = cv2.resize(img, (self.w,self.h), 0, 0, cv2.INTER_CUBIC)
        lbl = cv2.resize(lbl, (self.w,self.h), 0, 0, cv2.INTER_NEAREST)
        
        return img, lbl


class ToNDArray():
    def __call__(self, img, lbl):
        img = mx.nd.array(img) #TODO: dtype
        lbl = mx.nd.array(lbl) #TODO: dtype      
        return img, lbl

# 颜色与亮度增强
class Color_augmentation():
    def __call__(self, img, lbl):
        aug = mx.image.BrightnessJitterAug(brightness=0.4)
        aug_base = aug(img.astype('float32'))
        aug_base = aug_base.clip(0,255)
        # Add more color augmentations here...
        return img, lbl

# 归一化
class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, lbl):
        img = mx.image.color_normalize(img.astype('float32'), self.mean, self.std)
        img = mx.nd.transpose(img, (2, 0, 1))
        lbl = lbl.astype('int32').asnumpy()
        lbl = mx.nd.array(lbl)
        return img, lbl

#组合增强函数
class Compose:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, img, lbl):
        for t in self.trans:
            img, lbl = t(img, lbl)
        return img, lbl