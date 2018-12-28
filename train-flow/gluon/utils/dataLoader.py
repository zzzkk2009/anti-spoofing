# conding:utf-8

from mxnet.gluon.data import dataset
import mxnet as mx

import numpy as np
import os
import collections

import cv2

## dataLoder for classify
class DatasetForClassify(dataset.Dataset):
    """
    A dataset for loading images  stored as `xyz.jpg` and `xyz_mask.png`.
    Parameters
    ----------
    root : str
        Path to root directory.
    transform : callable, default None
        A function that takes data and label and transforms them:
    ::
        transform = lambda data, label: (data.astype(np.float32)/255, label)
    
    input_size: input_size format = (W, H), NOT as MXnet image (H,W)!
    
    Note: If you want to show the images that youeant to read, Please Commented the 
    color normalize function!
    """
    def __init__(self, root, transform=None, input_size=(360,640)):
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)
        self.mean = mx.nd.array([0.485, 0.456, 0.406])
        self.std = mx.nd.array([0.229, 0.224, 0.225])
        self._input_size = input_size
#         self._resize = isResize
    
    def _list_images(self, root):
        images = collections.defaultdict(dict)
        for filename in sorted(os.listdir(root)):
            name, ext = os.path.splitext(filename)
            mask_flag = name.endswith("_mask")
            if ext.lower() not in self._exts:
                continue
            if not mask_flag:
                images[name]["base"] = filename
            else:
                name = name[:-5] # to remove '_mask'
                images[name]["mask"] = filename
        self._image_list = list(images.values())

    def __getitem__(self, idx):
        assert 'base' in self._image_list[idx], "Couldn't find base image for: " + self._image_list[idx]["mask"]
        #read image
        base_filepath = os.path.join(self._root, self._image_list[idx]["base"])
        base = mx.image.imread(base_filepath)
#         base = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         if self._resize:
#             base = base.asnumpy()
#             base = cv2.resize(base,self._input_size)
#             base = nd.array(base)
#         base = color_augmentation(base)
#         base = image.color_normalize(base, self.mean, self.std)
# #         base = normalize_image(base)
#         base = base.transpose((2,0,1))
    
        #read mask
        assert 'mask' in self._image_list[idx], "Couldn't find mask image for: " + self._image_list[idx]["base"]
        mask_filepath = os.path.join(self._root, self._image_list[idx]["mask"])
        mask = mx.image.imread(mask_filepath)
        mask = mask[:,:,0]
#         if self._resize:
#             mask = mask.asnumpy()
#             mask = cv2.resize(mask,self._input_size,interpolation=cv2.INTER_NEAREST)
            
#         mask = mask.astype('int32')
#         mask = nd.array(mask)
        if self._transform is not None:
            return self._transform(base, mask)
        else:
            return base, mask

    def __len__(self):
        return len(self._image_list)