# -*-coding:utf-8-*-
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np
from mxnet.gluon.data import dataset
import os
import warnings
import random
from mxnet import gpu
from mxnet.gluon.data.vision import datasets
from MyImageFolderDataset import transform_train, transform_test

class MyImageFolderDataset(dataset.Dataset):
    def __init__(self, root, label, flag=1, transform=None):
        self._root = os.path.expanduser(root) # 把path中包含的"~"和"~user"转换成用户目录
        self._flag = flag
        self._label = label
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root, self._label)

    def _list_images(self, root, label):  # label是一个list
        self.synsets = []
        self.synsets.append(root)
        self.items = []
        #file = open(label)
        #lines = file.readlines()
        #random.shuffle(lines)
        c = 0
        for line in label:
            cls = line.split() # filename %t label
            filename = cls[0]
            label = cls[1]
            # print(os.path.join(root, filename))
            if os.path.isfile(os.path.join(root, filename)):
                self.items.append((os.path.join(root, filename), float(label)))
                # print((os.path.join(root, filename), float(label)))
            else:
                print('what')
            c = c + 1
        print('the total image is ', c)

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)


def get_dataloader(root, train_list_file, val_list_file, batch_size, transform_train, transform_val):
    loader = gluon.data.DataLoader
    with open(train_list_file, 'r') as f:
        train_list = f.readlines()
        train_ds = MyImageFolderDataset(root, train_list, flag=1, transform=transform_train)

    with open(val_list_file, 'r') as f:
        val_list = f.readlines()
        val_ds = MyImageFolderDataset(root, val_list, flag=1, transform=transform_val)
    
    train_loader = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
    val_loader = loader(val_ds, batch_size, shuffle=False, last_batch='keep')
    return train_loader, val_loader
